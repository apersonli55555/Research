# Builds HWM_Research_Paper.pdf from the markdown (figures embedded). Run from this folder.
import markdown,base64,re,pathlib,subprocess,os
md=pathlib.Path('HWM_Research_Paper.md').read_text()
html=markdown.markdown(md,extensions=['extra'])
html=re.sub(r'src="(exhibits/[^"]+\.png)"',lambda m:'src="data:image/png;base64,'+base64.b64encode(pathlib.Path(m.group(1)).read_bytes()).decode()+'"',html)
css='''body{font-family:"Times New Roman",Times,serif;font-size:11.5pt;line-height:1.5;color:#111}
h1{font-size:19pt;line-height:1.25;margin:0 0 6pt} h2{font-size:13.5pt;margin:18pt 0 6pt;break-after:avoid}
p{margin:0 0 8pt;text-align:justify} img{width:100%;margin:6pt 0 2pt} p:has(img){break-inside:avoid;break-after:avoid}
em{color:#333} hr{border:0;border-top:1px solid #999;margin:12pt 0}'''
pathlib.Path('_paper.html').write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{html}</body></html>')
js='''const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();
await p.goto('file://'+process.cwd()+'/_paper.html');await p.pdf({path:'HWM_Research_Paper.pdf',format:'Letter',displayHeaderFooter:true,headerTemplate:'<span></span>',
footerTemplate:'<div style="font-size:8px;width:100%;text-align:center;color:#666">HWM research paper | data as of Nov 6, 2026 close | page <span class=pageNumber></span> of <span class=totalPages></span></div>',
margin:{top:'0.9in',bottom:'0.9in',left:'0.95in',right:'0.95in'}});await b.close();})()'''
subprocess.run(['node','-e',js],check=True); os.remove('_paper.html'); print('PDF written')
