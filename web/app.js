const form=document.querySelector("#batch-form");
const list=document.querySelector("#pairs");
const template=document.querySelector("#pair-template");
const addButton=document.querySelector("#add-pair");
const submitButton=document.querySelector("#submit");
const results=document.querySelector("#results");
const progress=document.querySelector("#progress");
const maxPairs=100;

function refreshRows(){
  [...list.querySelectorAll(".pair-card")].forEach((card,index)=>{
    card.querySelector(".pair-number").textContent=`${index+1}.`;
    card.querySelector(".remove").disabled=list.children.length===1;
  });
  addButton.disabled=list.children.length>=maxPairs;
}

function addPair(){
  if(list.children.length>=maxPairs)return;
  const card=template.content.firstElementChild.cloneNode(true);
  card.querySelector(".remove").addEventListener("click",()=>{card.remove();refreshRows();});
  for(const [inputSelector,previewSelector] of [[".qr-file",".qr-preview"],[".house-file",".house-preview"]]){
    const input=card.querySelector(inputSelector),preview=card.querySelector(previewSelector);
    input.addEventListener("change",()=>{
      const file=input.files[0];
      if(!file){preview.removeAttribute("src");preview.classList.add("hidden");return;}
      preview.src=URL.createObjectURL(file);preview.classList.remove("hidden");
    });
  }
  list.append(card);refreshRows();
}

function dataUrl(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error(`Could not read ${file.name}`));reader.readAsDataURL(file);});}

function showResults(items){
  results.replaceChildren();
  for(const item of items){
    const card=document.createElement("article");
    let kind="manual",title="REQUIRED MANUAL REVIEW",detail="No status change was made.";
    if(item.success===false){kind="failed";title=`PAIR ${Number(item.index)+1} — INPUT/PROCESSING ERROR`;detail=item.error||"Could not process this pair.";}
    else if(item.QRStatus===1){kind="approved";title=`PAIR ${Number(item.index)+1} — APPROVE`;detail="The plate matched and inspector accepted the reference image.";}
    else if(item.QRStatus===0){kind="rejected";title=`PAIR ${Number(item.index)+1} — REJECT`;detail=item.Remark||"The pair did not pass inspection.";}
    else if(item.houseId!==undefined&&item.houseId!==null){title=`PAIR ${Number(item.index)+1} — REQUIRED MANUAL REVIEW (House ${item.houseId})`;}
    card.className=`result-card ${kind}`;
    const heading=document.createElement("strong");heading.textContent=title;
    const text=document.createElement("span");text.textContent=detail;
    card.append(heading,text);results.append(card);
  }
}

addButton.addEventListener("click",addPair);
form.addEventListener("submit",async event=>{
  event.preventDefault();
  const cards=[...list.querySelectorAll(".pair-card")];
  submitButton.disabled=true;addButton.disabled=true;submitButton.textContent="Processing…";
  progress.textContent=`Sending ${cards.length} pair${cards.length===1?"":"s"} to the matcher…`;
  results.replaceChildren();
  try{
    const items=await Promise.all(cards.map(async card=>{
      const qr=card.querySelector(".qr-file").files[0],house=card.querySelector(".house-file").files[0];
      if(!qr||!house)throw new Error("Select both a reference QR image and a property image for every pair.");
      const item={QRCodeImage:await dataUrl(qr),HouseImage:await dataUrl(house)};
      const houseId=card.querySelector(".house-id").value.trim();if(houseId)item.houseId=houseId;
      return item;
    }));
    const response=await fetch("/api/inspect/batch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({items})});
    const payload=await response.json();
    if(!response.ok)throw new Error(payload.detail?.map?.(e=>e.msg).join("; ")||payload.detail||`Request failed (${response.status})`);
    showResults(payload.results||[]);progress.textContent=`Processed ${payload.count} pair${payload.count===1?"":"s"}.`;
  }catch(error){
    const card=document.createElement("article");card.className="result-card failed";
    const heading=document.createElement("strong");heading.textContent="Could not check images";
    const detail=document.createElement("span");detail.textContent=error.message;card.append(heading,detail);results.append(card);
    progress.textContent="Correct the input and try again.";
  }finally{
    submitButton.disabled=false;submitButton.textContent="Check images";refreshRows();
  }
});

addPair();
