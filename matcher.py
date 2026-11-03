import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class HybridImageMatcher:
    # Calibrated against the uploaded distant-plate examples: tolerate lighting
    # and perspective changes while remaining above tested different-wall pairs.
    BACKGROUND_SIMILARITY_THRESHOLD = 0.55

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Grounding DINO is only a fallback. Delay loading it until template
        # matching misses, and use cached files only so offline runs stay usable.
        self.processor = None
        self.model = None
        self.qr_detector = cv2.QRCodeDetector()
        self.orb = cv2.ORB_create(nfeatures=8000, scaleFactor=1.15, nlevels=10, edgeThreshold=8, fastThreshold=7)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _find_qr_boxes(self, rgb_image):
        """Locate QR symbols without reading or comparing their payloads."""
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        boxes = []
        for scale in (1.0, 1.5, 2.0, 2.5, 3.0):
            # A fresh detector per scale avoids OpenCV retaining failed decode state.
            detector = cv2.QRCodeDetector()
            scaled = gray if scale == 1.0 else cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            points = None
            try:
                ok_multi, multi_points = detector.detectMulti(scaled)
                if ok_multi and multi_points is not None:
                    points = multi_points
            except cv2.error:
                pass
            try:
                ok, single_points = detector.detect(scaled)
                if points is None and ok and single_points is not None:
                    points = single_points
            except cv2.error:
                pass
            try:
                # Use returned corner coordinates only; decoded text is discarded.
                _, decoded_points, _ = detector.detectAndDecode(scaled)
                if points is None and decoded_points is not None:
                    points = decoded_points
            except cv2.error:
                pass
            if points is not None:
                point_array = np.asarray(points, dtype=np.float32)
                if point_array.ndim == 2:
                    point_array = point_array[None, ...]
                for quad in point_array:
                    coords = quad.reshape(-1, 2) / scale
                    x1, y1 = np.floor(coords.min(axis=0)).astype(int)
                    x2, y2 = np.ceil(coords.max(axis=0)).astype(int)
                    boxes.append([x1, y1, x2, y2])
        return boxes

    @staticmethod
    def _plate_crop_without_qr(ref_rgb):
        """Find a likely light plate region when QR corner detection fails."""
        h, w = ref_rgb.shape[:2]
        hsv = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2HSV)
        # Use connected bright regions, but avoid merging them through a large
        # morphology kernel (walls and UI overlays otherwise swallow the plate).
        mask = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([180, 115, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            ratio = cw / max(ch, 1)
            if area >= max(100, int(w * h * .0015)) and 1.15 <= ratio <= 5.0:
                # QR plates have many alternating dark/light cells. Favor regions
                # with substantial edge detail while avoiding the low-texture wall.
                gray = cv2.cvtColor(ref_rgb[y:y+ch, x:x+cw], cv2.COLOR_RGB2GRAY)
                edge_density = cv2.Canny(gray, 50, 140).mean()
                candidates.append((edge_density, area, x, y, cw, ch))
        if not candidates:
            return None
        _, _, x, y, cw, ch = max(candidates)
        px, py = max(4, int(cw * .08)), max(4, int(ch * .08))
        return (max(0, x-px), max(0, y-py), min(w, x+cw+px), min(h, y+ch+py))

    @staticmethod
    def _bright_plate_boxes(rgb_image):
        """Find candidate bright rectangular plates in a property scene."""
        h, w = rgb_image.shape[:2]
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([180, 115, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            ratio = cw / max(ch, 1)
            if area >= max(80, int(w * h * .00015)) and 1.15 <= ratio <= 5.0:
                boxes.append((x, y, x + cw, y + ch))
        return boxes

    @staticmethod
    def _background_similarity(ref_rgb, ref_box, scene_rgb, scene_box):
        """Compare color and surface texture surrounding two detected QR symbols."""
        def ring_signature(image, box):
            x1, y1, x2, y2 = box
            x1 = max(0, min(image.shape[1] - 1, int(x1)))
            y1 = max(0, min(image.shape[0] - 1, int(y1)))
            x2 = max(x1 + 1, min(image.shape[1], int(x2)))
            y2 = max(y1 + 1, min(image.shape[0], int(y2)))
            qr_w, qr_h = max(1, x2 - x1), max(1, y2 - y1)
            margin_x, margin_y = int(qr_w * 2.0), int(qr_h * 2.0)
            ox1, oy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
            ox2, oy2 = min(image.shape[1], x2 + margin_x), min(image.shape[0], y2 + margin_y)
            patch = image[oy1:oy2, ox1:ox2]
            mask = np.ones(patch.shape[:2], dtype=np.uint8) * 255
            ix1 = max(0, x1 - ox1 - int(qr_w * .15))
            iy1 = max(0, y1 - oy1 - int(qr_h * .15))
            ix2 = min(patch.shape[1], x2 - ox1 + int(qr_w * .15))
            iy2 = min(patch.shape[0], y2 - oy1 + int(qr_h * .15))
            mask[iy1:iy2, ix1:ix2] = 0
            if cv2.countNonZero(mask) < 20:
                return None
            lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
            color_hist = cv2.calcHist([lab], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(color_hist, color_hist, alpha=1, norm_type=cv2.NORM_L1)
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            valid = mask > 0
            texture_hist, _ = np.histogram(
                angle[valid] % 180, bins=12, range=(0, 180), weights=magnitude[valid]
            )
            texture_hist = texture_hist.astype(np.float32)
            texture_hist /= max(float(texture_hist.sum()), 1e-6)
            mean_lab = np.asarray(cv2.mean(lab, mask=mask)[:3], dtype=np.float32)
            return color_hist, texture_hist, mean_lab

        ref_sig = ring_signature(ref_rgb, ref_box)
        scene_sig = ring_signature(scene_rgb, scene_box)
        if ref_sig is None or scene_sig is None:
            return 0.0
        color_similarity = 1.0 - cv2.compareHist(ref_sig[0], scene_sig[0], cv2.HISTCMP_BHATTACHARYYA)
        texture_similarity = 1.0 - cv2.compareHist(
            ref_sig[1], scene_sig[1], cv2.HISTCMP_BHATTACHARYYA
        )
        mean_color_distance = float(np.linalg.norm(ref_sig[2] - scene_sig[2]))
        mean_color_similarity = max(0.0, 1.0 - mean_color_distance / 100.0)
        similarity = float(.55 * color_similarity + .2 * texture_similarity + .25 * mean_color_similarity)
        # When the two QR crops are similarly sized, compare surface detail too.
        # This catches smooth-vs-rough walls that have similar average colors and
        # gradient directions. Skip this term for distant plates because blur and
        # downsampling naturally erase fine texture.
        ref_scale = (ref_box[2] - ref_box[0], ref_box[3] - ref_box[1])
        scene_scale = (scene_box[2] - scene_box[0], scene_box[3] - scene_box[1])
        scale_x = scene_scale[0] / max(1, ref_scale[0])
        scale_y = scene_scale[1] / max(1, ref_scale[1])
        if .7 <= scale_x <= 1.4 and .7 <= scale_y <= 1.4:
            ref_edges = cv2.Canny(cv2.GaussianBlur(
                cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0
            ), 40, 110)
            scene_edges = cv2.Canny(cv2.GaussianBlur(
                cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0
            ), 40, 110)

            def ring_edge_density(edges, box):
                x1, y1, x2, y2 = map(int, box)
                qw, qh = max(1, x2 - x1), max(1, y2 - y1)
                ox1, oy1 = max(0, x1 - int(qw * 2)), max(0, y1 - int(qh * 2))
                ox2, oy2 = min(edges.shape[1], x2 + int(qw * 2)), min(edges.shape[0], y2 + int(qh * 2))
                patch = edges[oy1:oy2, ox1:ox2]
                mask = np.ones(patch.shape[:2], dtype=bool)
                ix1 = max(0, x1 - ox1 - int(qw * .15))
                iy1 = max(0, y1 - oy1 - int(qh * .15))
                ix2 = min(patch.shape[1], x2 - ox1 + int(qw * .15))
                iy2 = min(patch.shape[0], y2 - oy1 + int(qh * .15))
                mask[iy1:iy2, ix1:ix2] = False
                return float(np.mean(patch[mask] > 0)) if np.any(mask) else 0.0

            ref_density = ring_edge_density(ref_edges, ref_box)
            scene_density = ring_edge_density(scene_edges, scene_box)
            density_similarity = min(ref_density, scene_density) / max(ref_density, scene_density, 1e-6)
            similarity = .75 * similarity + .25 * density_similarity
        return float(similarity)

    @staticmethod
    def _map_context_box(template_box, scene_box, qr_box):
        """Project a QR box within a context crop onto a matched scene crop."""
        tx1, ty1, tx2, ty2 = template_box
        sx1, sy1, sx2, sy2 = scene_box
        scale_x = (sx2 - sx1) / max(1, tx2 - tx1)
        scale_y = (sy2 - sy1) / max(1, ty2 - ty1)
        qx1, qy1, qx2, qy2 = qr_box
        return [
            round(sx1 + (qx1 - tx1) * scale_x),
            round(sy1 + (qy1 - ty1) * scale_y),
            round(sx1 + (qx2 - tx1) * scale_x),
            round(sy1 + (qy2 - ty1) * scale_y),
        ]

    def _region_detections(self, image, prompt):
        if self.processor is None or self.model is None:
            self.processor = AutoProcessor.from_pretrained(
                "IDEA-Research/grounding-dino-tiny", local_files_only=True
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny", local_files_only=True
            ).to(self.device)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.18, text_threshold=0.15,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results.get("text_labels", [""] * len(boxes))
        return [
            {"box": box, "score": float(score), "label": str(label).casefold()}
            for box, score, label in zip(boxes, scores, labels)
        ]

    def _regions(self, image, prompt):
        """Compatibility helper returning only detected boxes."""
        return np.asarray([d["box"] for d in self._region_detections(image, prompt)], dtype=np.float32)

    def _match_region(self, ref_gray, scene_gray, offset_x=0, offset_y=0, min_inliers=12):
        kp1, des1 = self.orb.detectAndCompute(ref_gray, None)
        kp2, des2 = self.orb.detectAndCompute(scene_gray, None)
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return None
        pairs = self.matcher.knnMatch(des1, des2, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.72 * n.distance]
        if len(good) < min_inliers:
            return None
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = inliers / max(1, len(good))
        if homography is None or inliers < min_inliers or inlier_ratio < 0.5:
            return None
        h, w = ref_gray.shape
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        x1, y1 = np.floor(projected.min(axis=0)).astype(int)
        x2, y2 = np.ceil(projected.max(axis=0)).astype(int)
        return inliers, inlier_ratio, [int(x1 + offset_x), int(y1 + offset_y), int(x2 + offset_x), int(y2 + offset_y)]

    def _template_search(self, ref_gray, scene_gray, top_k=12):
        """Return several likely locations using normalized correlation at multiple scales."""
        rh, rw = ref_gray.shape
        sh, sw = scene_gray.shape
        if rh < 12 or rw < 12 or sh < 12 or sw < 12:
            return []
        # Matching the grayscale and edge images helps with exposure and color changes.
        # Remove low-frequency illumination/background differences. This matters when
        # the close-up contains wall around the plate, but the plate is tiny at distance.
        ref_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ref_gray)
        scene_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(scene_gray)
        ref_edge = cv2.Canny(cv2.GaussianBlur(ref_gray, (3, 3), 0), 35, 110)
        scene_edge = cv2.Canny(cv2.GaussianBlur(scene_gray, (3, 3), 0), 45, 130)
        candidates = []
        # Search small distant plates too; the caller also enforces a stronger
        # candidate score so isolated module fragments do not pass.
        for scale in np.geomspace(0.04, 1.8, 34):
            tw, th = int(round(rw * scale)), int(round(rh * scale))
            if tw < 12 or th < 12 or tw > sw or th > sh:
                continue
            template = cv2.resize(ref_gray, (tw, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            templ_edge = cv2.resize(ref_edge, (tw, th), interpolation=cv2.INTER_NEAREST)
            for image, templ in ((scene_gray, template), (scene_edge, templ_edge)):
                if float(templ.std()) < 2 or float(image.std()) < 2:
                    continue
                scores = cv2.matchTemplate(image, templ, cv2.TM_CCOEFF_NORMED)
                work = scores.copy()
                for _ in range(top_k):
                    _, score, _, loc = cv2.minMaxLoc(work)
                    if score < 0:
                        break
                    candidates.append((float(score), [loc[0], loc[1], loc[0] + tw, loc[1] + th]))
                    # Suppress neighboring peaks at this scale so one location
                    # cannot dominate all candidate slots.
                    x, y = loc
                    pad_x, pad_y = max(2, tw // 3), max(2, th // 3)
                    work[max(0, y-pad_y):min(work.shape[0], y+pad_y+1),
                         max(0, x-pad_x):min(work.shape[1], x+pad_x+1)] = -1
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0], reverse=True)
        unique = []
        for score, box in candidates:
            x1, y1, x2, y2 = box
            if any(
                abs(x1 - ub[0]) < max(4, (x2 - x1) // 3)
                and abs(y1 - ub[1]) < max(4, (y2 - y1) // 3)
                and abs((x2 - x1) - (ub[2] - ub[0])) < max(4, (x2 - x1) // 4)
                for _, ub in unique
            ):
                continue
            unique.append((score, box))
            if len(unique) >= top_k:
                break
        return unique

    @staticmethod
    def _sharpness(gray):
        if gray.size == 0:
            return 0.0
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @classmethod
    def _normalized_patch_sharpness(cls, gray_patch, target_size=96):
        if gray_patch.size == 0:
            return 0.0
        resized = cv2.resize(gray_patch, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        return cls._sharpness(resized)

    def verify_presence(self, ref_img, property_img, target_prompt="plate . sign . house number ."):
        """Return ACCEPT/REJECT evidence or a manual-review decision if unclear."""
        if not isinstance(ref_img, Image.Image):
            ref_img = Image.open(ref_img)
        if not isinstance(property_img, Image.Image):
            property_img = Image.open(property_img)
        ref_img = ref_img.convert("RGB")
        property_img = property_img.convert("RGB")
        ref_rgb = np.asarray(ref_img)
        prop_rgb = np.asarray(property_img)
        ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
        prop_gray = cv2.cvtColor(prop_rgb, cv2.COLOR_RGB2GRAY)

        # A blurry reference QR or a globally unusable property photo cannot
        # support an automatic decision. Stop before matching/inspection.
        reference_qrs = self._find_qr_boxes(ref_rgb)
        if reference_qrs:
            x1, y1, x2, y2 = map(int, reference_qrs[0])
            qr_gray = ref_gray[max(0, y1):min(ref_gray.shape[0], y2), max(0, x1):min(ref_gray.shape[1], x2)]
            ref_sharpness = self._normalized_patch_sharpness(qr_gray)
            if min(qr_gray.shape[:2], default=0) < 24 or ref_sharpness < 700:
                return {
                    "contains_reference": False,
                    "needs_manual_review": True,
                    "detection_method": "manual_review",
                    "manual_review_reason": "The reference QR plate image is too blurry or too small to verify reliably.",
                    "reference_sharpness": ref_sharpness,
                }
        elif min(ref_img.size) < 220 or self._sharpness(ref_gray) < 45:
            return {
                "contains_reference": False,
                "needs_manual_review": True,
                "detection_method": "manual_review",
                "manual_review_reason": "The reference image is too unclear to locate the QR plate reliably.",
            }

        if self._sharpness(prop_gray) < 12:
            return {
                "contains_reference": False,
                "needs_manual_review": True,
                "detection_method": "manual_review",
                "manual_review_reason": "The property image is too blurry to verify the plate and its surroundings.",
                "property_sharpness": self._sharpness(prop_gray),
            }

        result = self._verify_presence_core(ref_img, property_img, target_prompt)
        if result.get("recognition_uncertain") or result.get("detection_method") in {
            "QR symbol not localized; use clearer original images",
            "QR candidate found but background could not be validated",
        }:
            result.update({
                "contains_reference": False,
                "needs_manual_review": True,
                "detection_method": "manual_review",
                "manual_review_reason": result.get(
                    "manual_review_reason",
                    "The matcher could not confidently locate the QR plate. Please confirm the plate and its nearby background.",
                ),
            })
        return result

    def _verify_presence_core(self, ref_img, property_img, target_prompt="plate . sign . house number ."):
        if not isinstance(ref_img, Image.Image):
            ref_img = Image.open(ref_img)
        if not isinstance(property_img, Image.Image):
            property_img = Image.open(property_img)
        ref_img = ref_img.convert("RGB")
        property_img = property_img.convert("RGB")
        # Keep matching fast for large camera photos. Scale final coordinates back
        # to the uploaded property's resolution before returning them.
        max_side = 1400
        property_scale = min(1.0, max_side / max(property_img.size))
        if property_scale < 1.0:
            match_property = property_img.resize(
                (round(property_img.width * property_scale), round(property_img.height * property_scale)),
                Image.Resampling.LANCZOS,
            )
        else:
            match_property = property_img
        ref = np.asarray(ref_img)
        prop = np.asarray(match_property)
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
        prop_gray = cv2.cvtColor(prop, cv2.COLOR_RGB2GRAY)
        height, width = prop_gray.shape
        reference_qrs = self._find_qr_boxes(ref)

        # First locate the QR plate candidate using its visual details. Then
        # compare the background only around that candidate in the property photo.
        # QR payloads / printed plate IDs are deliberately not read or compared.
        ref_h, ref_w = ref_gray.shape
        plate_crop = ref_gray
        plate_box = (0, 0, ref_w, ref_h)
        if reference_qrs:
            # Isolate the detected QR symbol for the first-stage location search.
            # Background is evaluated only after this QR candidate is projected
            # into the property image below.
            x1, y1, x2, y2 = reference_qrs[0]
            qr_w, qr_h = max(1, x2 - x1), max(1, y2 - y1)
            margin_x, margin_y = max(2, int(qr_w * 0.06)), max(2, int(qr_h * 0.06))
            x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
            x2, y2 = min(ref_w, x2 + margin_x), min(ref_h, y2 + margin_y)
            plate_crop = ref_gray[y1:y2, x1:x2]
            plate_box = (x1, y1, x2, y2)
        else:
            inferred_box = self._plate_crop_without_qr(ref)
            if inferred_box:
                x1, y1, x2, y2 = inferred_box
                plate_crop = ref_gray[y1:y2, x1:x2]
                plate_box = inferred_box

        # Locate a candidate plate before evaluating the background around it.
        if reference_qrs:
            qr_match = self._match_region(plate_crop, prop_gray)
            if qr_match:
                inliers, inlier_ratio, bbox = qr_match
                x1, y1, x2, y2 = bbox
                projected_qr = self._map_context_box(plate_box, bbox, reference_qrs[0])
                projected_background = self._background_similarity(
                    ref, reference_qrs[0], prop, projected_qr
                )
                if projected_background >= self.BACKGROUND_SIMILARITY_THRESHOLD:
                    return {
                        "contains_reference": True,
                        "bbox": [
                            max(0, round(x1 / property_scale)), max(0, round(y1 / property_scale)),
                            min(property_img.width, round(x2 / property_scale)),
                            min(property_img.height, round(y2 / property_scale)),
                        ],
                        "match_score": inlier_ratio,
                        "background_similarity": projected_background,
                        "feature_inliers": inliers,
                        "candidate_qr_box": projected_qr,
                        "detection_method": "QR features and surrounding background",
                    }
                # A strong geometric QR match with a failed background check is
                # a decisive rejection. Do not let a weaker repeated pattern or
                # watermark elsewhere in the photo override the same candidate.
                return {
                    "contains_reference": False,
                    "detection_method": "QR background context mismatch",
                    "background_similarity": projected_background,
                    "feature_inliers": inliers,
                    "match_score": inlier_ratio,
                    "candidate_qr_box": projected_qr,
                }
        # Search every location at multiple scales. Full sliding matching avoids
        # missing a small, distant QR that falls between fixed grid sample points.
        template_matches = self._template_search(plate_crop, prop_gray)
        # This score only nominates a QR candidate. Background context below is a
        # separate required check, so a weak visual candidate cannot pass alone.
        template_threshold = 0.60
        if template_matches and template_matches[0][0] >= template_threshold:
            best_score = template_matches[0][0]
            accepted = None
            best_background = None
            best_background_box = None
            for score, bbox in template_matches:
                if score < template_threshold:
                    break
                if not reference_qrs:
                    continue
                projected_qr = self._map_context_box(plate_box, bbox, reference_qrs[0])
                background_score = self._background_similarity(
                    ref, reference_qrs[0], prop, projected_qr
                )
                if best_background is None or background_score > best_background:
                    best_background = background_score
                    best_background_box = projected_qr
                if background_score >= self.BACKGROUND_SIMILARITY_THRESHOLD:
                    accepted = (score, bbox, background_score, projected_qr)
                    break
            if not reference_qrs:
                return {
                    "contains_reference": False,
                    "detection_method": "QR candidate found but background could not be validated",
                    "best_template_score": best_score,
                }
            if accepted is None:
                return {
                    "contains_reference": False,
                    "detection_method": "QR background context mismatch",
                    "background_similarity": best_background,
                    "best_template_score": best_score,
                    "candidate_qr_box": best_background_box,
                }
            score, bbox, projected_background, candidate_qr_box = accepted
            if reference_qrs and plate_box:
                # Report the whole plate area, not only the thresholded white surface.
                scale_x = (bbox[2] - bbox[0]) / max(1, plate_crop.shape[1])
                scale_y = (bbox[3] - bbox[1]) / max(1, plate_crop.shape[0])
                bbox = [
                    round(bbox[0] - plate_box[0] * scale_x),
                    round(bbox[1] - plate_box[1] * scale_y),
                    round(bbox[2] + (ref_w - plate_box[2]) * scale_x),
                    round(bbox[3] + (ref_h - plate_box[3]) * scale_y),
                ]
            x1, y1, x2, y2 = bbox
            return {
                "contains_reference": True,
                "bbox": [
                    max(0, round(x1 / property_scale)), max(0, round(y1 / property_scale)),
                    min(property_img.width, round(x2 / property_scale)),
                    min(property_img.height, round(y2 / property_scale)),
                ],
                "match_score": score,
                "background_similarity": projected_background if reference_qrs else None,
                "candidate_qr_box": candidate_qr_box,
                "detection_method": "Localized plate template matching",
            }
        # If corner detection fails (common for tiny, blurry plates), nominate
        # bright plate-like scene regions first, then compare them with the
        # reference crop and validate their local background. This still avoids
        # accepting on printed IDs or generic wall similarity alone.
        if reference_qrs and target_prompt.strip():
            try:
                labels = ["qr code"] + [
                    part.strip() for part in target_prompt.split(".") if part.strip().casefold() != "qr code"
                ]
                dino_prompt = " . ".join(labels) + " ."
                detections = self._region_detections(match_property, dino_prompt)
                plausible = []
                qr_detections = [
                    d for d in detections if "qr code" in d["label"] and d["score"] >= .18
                ]
                for detection in detections:
                    if "qr code" not in detection["label"] or detection["score"] < .28:
                        continue
                    x1, y1, x2, y2 = np.rint(detection["box"]).astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    box_w, box_h = x2 - x1, y2 - y1
                    if box_w < 8 or box_h < 8:
                        continue
                    if box_w / width > .12 or box_h / height > .10:
                        continue
                    ratio = box_w / max(1, box_h)
                    if not .65 <= ratio <= 4.0:
                        continue
                    plausible.append((detection["score"], [x1, y1, x2, y2]))

                best_dino = None
                for detection_score, candidate_box in sorted(plausible, reverse=True):
                    if detection_score < .33:
                        return {
                            "contains_reference": False,
                            "recognition_uncertain": True,
                            "detection_method": "low-confidence QR candidate",
                            "manual_review_reason": "A possible QR plate was found, but the detector confidence is too low for an automatic decision.",
                            "match_score": detection_score,
                            "candidate_qr_box": candidate_box,
                        }
                    background_score = self._background_similarity(
                        ref, reference_qrs[0], prop, candidate_box
                    )
                    if best_dino is None or background_score > best_dino[0]:
                        best_dino = (background_score, detection_score, candidate_box)
                    if background_score >= self.BACKGROUND_SIMILARITY_THRESHOLD:
                        x1, y1, x2, y2 = candidate_box
                        return {
                            "contains_reference": True,
                            "bbox": [
                                max(0, round(x1 / property_scale)),
                                max(0, round(y1 / property_scale)),
                                min(property_img.width, round(x2 / property_scale)),
                                min(property_img.height, round(y2 / property_scale)),
                            ],
                            "match_score": detection_score,
                            "background_similarity": background_score,
                            "candidate_qr_box": candidate_box,
                            "detection_method": "Grounding DINO QR localization + surrounding background",
                        }
                if best_dino:
                    return {
                        "contains_reference": False,
                        "detection_method": "QR background context mismatch",
                        "background_similarity": best_dino[0],
                        "match_score": best_dino[1],
                        "candidate_qr_box": best_dino[2],
                    }
                if qr_detections:
                    return {
                        "contains_reference": False,
                        "recognition_uncertain": True,
                        "detection_method": "low-confidence QR candidate",
                        "manual_review_reason": "The QR detector found a possible plate, but could not localize it confidently enough to check the background.",
                        "match_score": max(d["score"] for d in qr_detections),
                    }
            except Exception:
                # Keep the fast path usable when model downloads/inference are unavailable.
                pass
        if not reference_qrs:
            return {
                "contains_reference": False,
                "detection_method": "QR symbol not localized; use clearer original images",
                "best_template_score": template_matches[0][0] if template_matches else None,
            }
        if template_matches and .40 <= template_matches[0][0] < template_threshold:
            return {
                "contains_reference": False,
                "recognition_uncertain": True,
                "detection_method": "weak QR visual candidate",
                "manual_review_reason": "A possible QR candidate scored too weakly for a reliable automatic decision.",
                "best_template_score": template_matches[0][0],
            }
        return {"contains_reference": False, "best_template_score": template_matches[0][0] if template_matches else None}
