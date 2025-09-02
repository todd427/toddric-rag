import hashlib
from typing import List, Dict
def simple_chunks(text:str,size:int=900,overlap:int=150,min_len:int=200)->List[Dict]:
    text=text.strip(); chunks=[]; i=0; ord_idx=0; n=len(text)
    while i<n:
        j=min(i+size,n); k=text.rfind('. ', i, j); k=j if (k==-1 or k<i+min_len) else k+1
        chunk=text[i:k].strip()
        if chunk: chunks.append({'ord':ord_idx,'text':chunk}); ord_idx+=1
        i=max(k-overlap, i+1)
    return chunks
def checksum(s:str)->str: return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]
