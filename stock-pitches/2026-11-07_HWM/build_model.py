import sys
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.comments import Comment
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter as L

OUT=sys.argv[1]
wb=Workbook()
F='Arial'
BLUE=Font(name=F,color='0000FF',size=10); BLK=Font(name=F,color='000000',size=10)
GRN=Font(name=F,color='008000',size=10); BOLD=Font(name=F,bold=True,size=10)
HDR=Font(name=F,bold=True,color='FFFFFF',size=10); TITLE=Font(name=F,bold=True,size=14)
ITAL=Font(name=F,italic=True,size=9,color='555555')
HFILL=PatternFill('solid',fgColor='1F3864'); YEL=PatternFill('solid',fgColor='FFFF00')
SUB=PatternFill('solid',fgColor='D9E1F2')
thin=Side(style='thin',color='999999'); TOP=Border(top=thin)
USD='$#,##0;($#,##0);"-"'; USD1='$#,##0.0;($#,##0.0);"-"'; USD2='$#,##0.00;($#,##0.00);"-"'
PCT='0.0%;(0.0%);"-"'; MULT='0.0"x"'; NUM='#,##0;(#,##0);"-"'; NUM1='#,##0.0;(#,##0.0);"-"'; NUM2='#,##0.00'

def put(ws,ref,val,font=None,fmt=None,fill=None,bold=False,note=None):
    c=ws[ref]; c.value=val
    if font is None:
        font = BLK if (isinstance(val,str) and val.startswith('=')) else BLUE
        if isinstance(val,str) and val.startswith('=') and '!' in val and not any(o in val[1:] for o in '+-*/(') : font=GRN
    if bold: font=Font(name=F,bold=True,color=font.color.rgb if font.color else '000000',size=10)
    c.font=font
    if fmt: c.number_format=fmt
    if fill: c.fill=fill
    if note: c.comment=Comment(note,'Analyst')
    return c
def label(ws,ref,text,bold=False,ital=False):
    c=ws[ref]; c.value=text; c.font=BOLD if bold else (ITAL if ital else BLK); return c
def header(ws,row,vals,start=1):
    for i,v in enumerate(vals):
        c=ws.cell(row=row,column=start+i,value=v); c.font=HDR; c.fill=HFILL
        c.alignment=Alignment(horizontal='center' if i else 'left')
def section(ws,row,text,ncol):
    for i in range(1,ncol+1): ws.cell(row=row,column=i).fill=SUB
    ws.cell(row=row,column=1,value=text).font=BOLD

SEC10K='HWM FY2025 10-K (filed 2026-02-12, accn 0000004281-26-000012)'
SEC10Q='HWM Q3 2026 10-Q (filed 2026-10-29, accn 0000004281-26-000036)'
SECQ4='HWM Q4 2025 earnings release, 8-K Ex.99.1 (filed 2026-02-12)'
SECQ3='HWM Q3 2026 earnings release, 8-K Ex.99.1 (filed 2026-10-29)'

# ================= README =================
ws=wb.active; ws.title='README'
ws.column_dimensions['A'].width=110
lines=[('Howmet Aerospace (NYSE: HWM): Financial Model',TITLE),
('Pricing date: Nov 6, 2026 close | Model date: Nov 7, 2026 | All $ in millions except per-share data',BOLD),
('',None),
('COLOR LEGEND',BOLD),
('Blue text = hard-coded input (historical data from SEC filings, or analyst assumption). Every input has a source in the adjacent column or a cell comment.',BLUE),
('Black text = formula.   Green text = direct link to another sheet.   Yellow fill = key judgment assumption: change these to run scenarios.',BLK),
('',None),
('SHEETS',BOLD),
('Inputs: market data, balance sheet, WACC, terminal value and valuation-weight assumptions',None),
('Historicals: FY2023-FY2025, 9M2025, 9M2026, and LTM (last twelve months to Sep 30, 2026) from 10-K/10-Q; segments; end markets',None),
('Model: 5-year projections FY2027E-FY2031E, plus FY2026E bridge year anchored to management guidance',None),
('DCF: unlevered free cash flow DCF (perpetuity-growth and exit-multiple terminal values), sensitivity tables, reverse DCF',None),
('Comps: trading comparables on LTM GAAP figures pulled from each company\'s latest 10-K/10-Q (SEC XBRL)',None),
('Summary: valuation football field, blended fair value, 12-month price target, recommendation',None),
('',None),
('DATA SOURCES (all fundamentals from SEC EDGAR primary filings)',BOLD),
(SEC10K+': https://www.sec.gov/Archives/edgar/data/4281/000000428126000012/hwm-20251231.htm',None),
(SEC10Q+': https://www.sec.gov/Archives/edgar/data/4281/000000428126000036/hwm-20260930.htm',None),
(SECQ3+': https://www.sec.gov/Archives/edgar/data/4281/000110465926119857/tm2628456d1_ex99-1.htm',None),
(SECQ4+': https://www.sec.gov/Archives/edgar/data/4281/000110465926013832/tm266060d1_ex99-1.htm',None),
('Peer data: SEC XBRL companyfacts API (data.sec.gov) for TDG, HEI, ATI, CRS, WWD: latest 10-K + 10-Q as of Nov 7, 2026',None),
('Share prices: closing prices for Nov 6, 2026 from stockanalysis.com historical price pages (retrieved Nov 7, 2026)',None),
('Risk-free rate: FRED series DGS10 (10-Year Treasury constant maturity), 4.84% on Nov 5, 2026 (latest available)',None),
('',None),
('IMPORTANT CAVEATS',BOLD),
('Beta (1.20) is set to match third-party published estimates (Yahoo 5Y monthly 1.19; Dividend Vision 1.21); it was NOT computed by the analyst. Equity risk premium (5.0%) is an analyst assumption.',None),
('Forecasts are the analyst\'s estimates. FY2026E/FY2027E revenue and FY2026E Adj. EBITDA are anchored to management guidance (Q3 2026 release).',None),
('This is an educational model, not investment advice.',ITAL),
]
for i,(t,f) in enumerate(lines,1):
    c=ws.cell(row=i,column=1,value=t)
    c.font=f if f else BLK; c.alignment=Alignment(wrap_text=True)

# ================= Inputs =================
ws=wb.create_sheet('Inputs')
for col,w in zip('ABCD',[46,14,10,95]): ws.column_dimensions[col].width=w
put(ws,'A1','Inputs & Assumptions',TITLE)
header(ws,3,['Item','Value','Unit','Source / rationale'])
I={}
rows=[
('MARKET DATA',None,None,None,None),
('price','Share price (Nov 6, 2026 close)',205.81,USD2,'stockanalysis.com HWM history page, close Nov 6, 2026 (retrieved Nov 7, 2026)'),
('shares','Basic shares outstanding (mm)',397.66049,NUM1,'10-Q cover page: 397,660,490 shares as of Oct 26, 2026'),
('dilut','Dilutive securities (mm)',2.0,NUM1,'Q3 2026 diluted (400mm) minus basic (398mm) weighted shares, 10-Q income statement'),
('mcap','Market capitalization','=B6*(B7+B8)',USD,'Price x diluted shares'),
('BALANCE SHEET (Sep 30, 2026)',None,None,None,None),
('cash','Cash and cash equivalents',507,USD,SEC10Q+', balance sheet'),
('cp','Short-term borrowings (commercial paper)',390,USD,SEC10Q+', balance sheet'),
('ltdc','Long-term debt due within one year',1,USD,SEC10Q+', balance sheet'),
('ltd','Long-term debt, less amount due within one year',4052,USD,SEC10Q+', balance sheet'),
('debt','Total debt','=SUM(B12:B14)',USD,'Ties to 10-Q XBRL DebtLongtermAndShorttermCombinedAmount = $4,443'),
('netdebt','Net debt','=B15-B11',USD,'Excludes pension deficit ($493 noncurrent) and leases ($178): see PM critique'),
('ev','Enterprise value (market)','=B9+B16',USD,'Market cap + net debt'),
('COST OF CAPITAL (WACC)',None,None,None,None),
('rf','Risk-free rate (10Y UST)',0.0484,PCT,'FRED DGS10, 4.84% on Nov 5, 2026'),
('erp','Equity risk premium',0.05,PCT,'ASSUMPTION: within the commonly used 4.5-5.5% range for the US market'),
('beta','Levered beta',1.20,NUM2,'Third-party estimates: Yahoo Finance 5Y monthly beta 1.19 (Mar 2026); Dividend Vision 1.212 (Sep 3, 2026); ValueRay 1.142. Not self-computed: verify with your own regression. WACC sensitivity on DCF sheet'),
('ke','Cost of equity (CAPM)','=B19+B21*B20',PCT,'CAPM: rf + beta x ERP'),
('kd','Pre-tax cost of debt','=Historicals!F15*4/3/B15',PCT,'9M2026 net interest expense ($142) annualized / total debt. Approximation'),
('tax','Tax rate (forecast)',0.21,PCT,'9M2026 effective rate was 19.1% (397/2078); Q3 2026 was 21.4%. Use 21%'),
('kdat','After-tax cost of debt','=B23*(1-B24)',PCT,''),
('we','Equity weight (market)','=B9/(B9+B15)',PCT,'Market-value weights'),
('wd','Debt weight','=B15/(B9+B15)',PCT,''),
('wacc','WACC','=B26*B22+B27*B25',PCT,''),
('TERMINAL VALUE & TIMING',None,None,None,None),
('g','Terminal (perpetual) growth rate',0.03,PCT,'ASSUMPTION: approx. long-run nominal GDP; aerospace grows faster but 2031 is not steady state'),
('exitm','Exit EV/EBITDA multiple (on FY2031E)',18.0,MULT,'ASSUMPTION: discount to today\'s peer median (see Comps) because growth will be slower by 2031'),
('valdate','Valuation date','=DATE(2026,11,6)','yyyy-mm-dd','Pricing date'),
('stub','Q4 2026E free cash flow (stub period)','=2000-Historicals!F32',USD,'FY2026 FCF guidance midpoint $2,000 (Q3 release) minus 9M2026 actual FCF'),
('GUIDANCE (Q3 2026 release, Oct 29, 2026)',None,None,None,None),
('g26rev','FY2026 revenue guidance (baseline)',10300,USD,SECQ3+'. Range $10,290-$10,310'),
('g26ebitda','FY2026 Adj. EBITDA guidance (baseline)',3330,USD,SECQ3+'. Range $3,325-$3,335'),
('g26eps','FY2026 Adj. EPS guidance (baseline)',5.50,USD2,SECQ3+'. Range $5.49-$5.51'),
('g26fcf','FY2026 free cash flow guidance (baseline)',2000,USD,SECQ3+'. Range $1,950-$2,050'),
('g27rev','FY2027 revenue guidance (approx.)',11500,USD,SECQ3+': "approximately $11.5 billion"'),
('VALUATION WEIGHTS (Summary sheet)',None,None,None,None),
('w1','Weight: DCF perpetuity growth',0.25,PCT,'ASSUMPTION'),
('w2','Weight: DCF exit multiple',0.25,PCT,'ASSUMPTION'),
('w3','Weight: Comps EV/EBITDA',0.25,PCT,'ASSUMPTION'),
('w4','Weight: Comps P/E',0.25,PCT,'ASSUMPTION'),
('thresh','Long/short threshold (+/- expected return)',0.15,PCT,'Rule: LONG if 12m upside > +15%, SHORT if < -15%, else NEUTRAL'),
]
r=5
for key,lab,val,fmt,src in rows:
    if lab is None:
        section(ws,r,key,4); r+=1; continue
    I[key]=f'Inputs!$B${r}'
    label(ws,f'A{r}',lab)
    fill = YEL if key in ('erp','beta','g','exitm','tax','w1','w2','w3','w4','thresh') else None
    put(ws,f'B{r}',val,fmt=fmt,fill=fill)
    label(ws,f'D{r}',src,ital=True)
    r+=1
# sanity: verify that hardcoded row refs match
assert I['price']=='Inputs!$B$6' and I['debt']=='Inputs!$B$15' and I['mcap']=='Inputs!$B$9', I
assert I['rf']=='Inputs!$B$19' and I['beta']=='Inputs!$B$21' and I['kd']=='Inputs!$B$23' and I['wacc']=='Inputs!$B$28', I

# ================= Historicals =================
ws=wb.create_sheet('Historicals')
ws.column_dimensions['A'].width=44
for c in 'BCDEFG': ws.column_dimensions[c].width=11
ws.column_dimensions['H'].width=70
put(ws,'A1','Historical Financials ($mm, except per share)',TITLE)
header(ws,3,['','FY2023','FY2024','FY2025','9M2025','9M2026','LTM Sep-26','Source'])
H={}
def hrow(r,key,lab,vals,fmt=USD,src='',ltm=True,bold=False):
    H[key]=r
    label(ws,f'A{r}',lab,bold=bold)
    for i,v in enumerate(vals):
        if v is None: continue
        put(ws,f'{"BCDEFG"[i]}{r}',v,fmt=fmt)
    if ltm is True: put(ws,f'G{r}',f'=D{r}+F{r}-E{r}',fmt=fmt)
    elif isinstance(ltm,str): put(ws,f'G{r}',ltm,fmt=fmt)
    label(ws,f'H{r}',src,ital=True)
IS=f'{SEC10K} & {SEC10Q}: income statement'
section(ws,4,'INCOME STATEMENT',8)
hrow(5,'rev','Sales',[6640,7430,8252,6084,7563],src=IS)
hrow(6,'cogs','Cost of goods sold (excl. D&A)',[4773,5119,5432,4020,4755],src=IS)
hrow(7,'sga','SG&A and other',[333,347,370,274,379],src=IS)
hrow(8,'rd','Research & development',[36,33,37,27,28],src=IS)
hrow(9,'da','Depreciation & amortization',[272,277,283,210,251],src=IS)
hrow(10,'restr','Restructuring and other charges (credits)',[23,21,84,-4,-91],src=IS+'. 9M2026 credit is mainly the gain on sale of the Savannah disk forging plant')
hrow(11,'oi','Operating income',[f'={c}5-SUM({c}6:{c}10)' for c in 'BCDEF'],src='Formula. Ties to reported: 1,203 / 1,633 / 2,046 / 1,557 / 2,241',bold=True)
hrow(12,'oicheck','  check vs. reported operating income',[f'={c}11-{v}' for c,v in zip('BCDEF',[1203,1633,2046,1557,2241])],fmt=NUM,src='Should be zero',ltm=False)
hrow(13,'ldr','Loss on debt redemption',[2,6,15,0,0],src=IS)
hrow(14,'other','Other expense, net',[8,62,40,33,21],src=IS)
hrow(15,'int','Interest expense, net',[218,182,151,114,142],src=IS)
hrow(16,'pbt','Income before income taxes',[f'={c}11-{c}13-{c}14-{c}15' for c in 'BCDEF'],src='Formula. Reported: 975 / 1,383 / 1,840 / 1,410 / 2,078',bold=True)
hrow(17,'taxx','Provision for income taxes',[210,228,332,274,397],src=IS)
hrow(18,'ni','Net income',[f'={c}16-{c}17' for c in 'BCDEF'],src='Formula. Reported: 765 / 1,155 / 1,508 / 1,136 / 1,681',bold=True)
hrow(19,'dsh','Diluted weighted shares (mm)',[416,410,406,406,402],fmt=NUM,src=IS,ltm=False)
hrow(20,'eps','Diluted EPS (reported)',[1.83,2.81,3.71,2.79,4.18],fmt=USD2,src=IS)
hrow(21,'etr','Effective tax rate',[f'={c}17/{c}16' for c in 'BCDEFG'],fmt=PCT,src='Formula',ltm=False)
section(ws,22,'NON-GAAP (company-defined)',8)
hrow(23,'aebitda','Adj. EBITDA excl. special items',[None,1914,2416,1763,2436],src=SECQ4+' (FY24, FY25); '+SECQ3+' (9M)')
hrow(24,'aebm','Adj. EBITDA margin',[None]+[f'={c}23/{c}5' for c in 'CDEF'],fmt=PCT,src='Formula',ltm='=G23/G5')
hrow(25,'oim','GAAP operating margin',[f'={c}11/{c}5' for c in 'BCDEF'],fmt=PCT,src='Formula',ltm='=G11/G5')
hrow(26,'growth','Revenue growth (YoY)',[None,'=C5/B5-1','=D5/C5-1',None,'=F5/E5-1'],fmt=PCT,src='Formula',ltm=False)
section(ws,27,'CASH FLOW',8)
CF=f'{SEC10K} & {SEC10Q}: cash flow statement'
hrow(28,'cfo','Cash from operations',[901,1298,1884,1230,1745],src=CF)
hrow(29,'capex','Capital expenditures',[219,321,453,329,345],src=CF)
hrow(30,'buyb','Share repurchases',[250,500,700,500,1100],src=CF)
hrow(31,'div','Dividends paid',[73,109,181,131,153],src=CF)
hrow(32,'fcf','Free cash flow (CFO - capex)',[f'={c}28-{c}29' for c in 'BCDEF'],src='Formula. Company FCF: FY25 $1,431; 9M26 $1,400 (ties)',bold=True)
hrow(33,'fcfconv','FCF conversion (FCF / net income)',[f'={c}32/{c}18' for c in 'BCDEFG'],fmt=PCT,src='Formula: above ~90-100% = high-quality earnings',ltm=False)
hrow(34,'capexpct','Capex % of sales',[f'={c}29/{c}5' for c in 'BCDEFG'],fmt=PCT,src='Formula',ltm=False)
hrow(35,'sbc','Stock-based compensation',[50,63,73,59,80],src=CF+'. Already expensed within Adj. EBITDA (not added back)')
section(ws,36,'WORKING CAPITAL (period end)',8)
BS=f'{SEC10K} & {SEC10Q}: balance sheet. 9M columns = Sep 30 balances'
hrow(37,'ar','Receivables from customers',[675,689,779,884,1116],src=BS,ltm=False)
hrow(38,'inv','Inventories',[1765,1840,1849,1975,2113],src=BS,ltm=False)
hrow(39,'ap','Accounts payable, trade',[982,948,845,957,969],src=BS,ltm=False)
hrow(40,'nwc','Trade working capital',[f'={c}37+{c}38-{c}39' for c in 'BCDEF'],src='Formula',ltm=False)
hrow(41,'nwcpct','Trade NWC % of (annualized) sales',['=B40/B5','=C40/C5','=D40/D5','=E40/(E5*4/3)','=F40/(F5*4/3)'],fmt=PCT,src='Formula; 9M sales annualized',ltm=False)
hrow(42,'dnwc','Change in NWC / change in sales',[None,'=(C40-B40)/(C5-B5)','=(D40-C40)/(D5-C5)',None,None],fmt=PCT,src='Formula: incremental working capital intensity',ltm=False)

# segments
section(ws,44,'SEGMENTS: third-party sales',8)
SG=f'{SEC10K} Note (segments) & {SEC10Q} Note D'
segs=[('Engine Products',[3266,3735,4320,3099,4077],[887,1150,1438,1023,1542]),
      ('Fastening Systems',[1349,1576,1745,1291,1690],[278,406,530,391,512]),
      ('Engineered Structures',[878,1065,1148,919,858],[113,166,243,199,200]),
      ('Forged Wheels',[1147,1054,1039,775,938],[309,287,296,217,264])]
r=45
for n,s,e in segs: hrow(r,'s_'+n,n,s,src=SG); r+=1
hrow(r,'segtot','Total',[f'=SUM({c}45:{c}48)' for c in 'BCDEF'],src='Formula; ties to consolidated sales',bold=True); r+=1
section(ws,r,'SEGMENTS: Segment Adjusted EBITDA',8); r+=1
e0=r
for n,s,e in segs: hrow(r,'e_'+n,n,e,src=SG); r+=1
hrow(r,'eseg','Total Segment Adj. EBITDA',[f'=SUM({c}{e0}:{c}{r-1})' for c in 'BCDEF'],src='Formula. Reported: 1,587 / 2,009 / 2,507 / 1,830 / 2,518',bold=True); r+=1
section(ws,r,'SEGMENTS: Segment Adj. EBITDA margin',8); r+=1
for i,(n,s,e) in enumerate(segs):
    hrow(r,'m_'+n,n,[f'={c}{e0+i}/{c}{45+i}' for c in 'BCDEF'],fmt=PCT,src='Formula',ltm=f'=G{e0+i}/G{45+i}'); r+=1
section(ws,r,'END MARKETS: revenue',8); r+=1
EM=f'{SEC10K} & {SEC10Q}: disaggregation of revenue by market'
m0=r
for n,v in [('Aerospace: Commercial',[3229,3871,4322,3185,4033]),('Aerospace: Defense',[1015,1164,1409,1042,1146]),
            ('Commercial Transportation',[1402,1308,1248,938,1102]),('Gas Turbines',[674,755,944,678,954]),('Other',[320,332,329,241,328])]:
    hrow(r,'em_'+n,n,v,src=EM); r+=1
hrow(r,'emtot','Total',[f'=SUM({c}{m0}:{c}{r-1})' for c in 'BCDEF'],src='Formula; ties to consolidated sales',bold=True); emtot=r; r+=1
section(ws,r,'END MARKETS: % of total',8); r+=1
for i in range(5):
    hrow(r,f'emp{i}',ws.cell(row=m0+i,column=1).value,[f'={c}{m0+i}/{c}{emtot}' for c in 'BCDEF'],fmt=PCT,src='Formula',ltm=f'=G{m0+i}/G{emtot}'); r+=1
section(ws,r,'END MARKETS: YoY growth (9M2026 vs 9M2025)',8); r+=1
for i in range(5):
    hrow(r,f'emg{i}',ws.cell(row=m0+i,column=1).value,[None,f'=C{m0+i}/B{m0+i}-1',f'=D{m0+i}/C{m0+i}-1',None,f'=F{m0+i}/E{m0+i}-1'],fmt=PCT,src='Formula',ltm=False); r+=1
ws.freeze_panes='B4'

# ================= Model =================
ws=wb.create_sheet('Model')
ws.column_dimensions['A'].width=40
for c in 'BCDEFGHI': ws.column_dimensions[c].width=11
ws.column_dimensions['J'].width=80
put(ws,'A1','Projections ($mm, except per share)',TITLE)
label(ws,'A2','FY2026E is a bridge year (9 months already reported, anchored to guidance). FY2027E-FY2031E are the 5 projection years used in the DCF.',ital=True)
cols=['FY2024A','FY2025A','FY2026E','FY2027E','FY2028E','FY2029E','FY2030E','FY2031E']
header(ws,4,['']+cols+['Notes'])
P=list('DEFGHI')  # forecast cols (2026E..2031E)
M={}
def mrow(r,key,lab,a24,a25,fc,fmt=USD,note='',bold=False,fill=None):
    M[key]=r; label(ws,f'A{r}',lab,bold=bold)
    if a24 is not None: put(ws,f'B{r}',a24,fmt=fmt)
    if a25 is not None: put(ws,f'C{r}',a25,fmt=fmt)
    for i,c in enumerate(P):
        v=fc(c,i) if callable(fc) else fc[i]
        if v is None: continue
        put(ws,f'{c}{r}',v,fmt=fmt,fill=fill if (isinstance(v,(int,float))) else None)
    label(ws,f'J{r}',note,ital=True)
prev=lambda c: chr(ord(c)-1)
section(ws,5,'DRIVERS (analyst assumptions in blue; yellow = key)',10)
mrow(6,'gr','Revenue growth (FY2028E+)',None,None,[None,None,0.09,0.08,0.07,0.06],fmt=PCT,fill=YEL,
     note='FY26E/FY27E revenue set to guidance (Inputs). Then decel 9%->6%: build-rate ramp continues but growth normalizes from 2026\'s ~25%')
mrow(7,'mg','Adj. EBITDA margin',None,None,[f'={I["g26ebitda"]}/{I["g26rev"]}',0.330,0.335,0.340,0.340,0.340],fmt=PCT,fill=YEL,
     note='FY26E = guidance ($3,330/$10,300 = 32.3%). Q3 2026 already 32.5%. Modest expansion, capped at 34%')
mrow(8,'dap','D&A % of sales',None,None,[0.034]*6,fmt=PCT,note='FY2025 3.4%, 9M2026 3.3% (incl. CAM acquired intangibles)')
mrow(9,'cxp','Capex % of sales',None,None,[0.05,0.055,0.055,0.055,0.05,0.05],fmt=PCT,fill=YEL,
     note='10-K: ~5% of sales in 2026. Q3 release: "elevated spend beyond 2027" for aero + gas turbine capacity')
mrow(10,'nwcp','Incremental NWC % of incremental sales',None,None,[0.12]*6,fmt=PCT,fill=YEL,
     note='History: 15.6% (FY24), 24.6% (FY25, incl. inventory build). 12% assumes some normalization: aggressive?')
mrow(11,'intr','Interest expense, net',None,None,[190,190,185,180,175,170],note='9M2026 $142 -> ~$190 run-rate post-CAM financing; slow decline')
mrow(12,'oth','Other expense, net',None,None,[28]*6,note='9M2026 $21 annualized')
mrow(13,'shc','Diluted share count change (YoY)',None,None,[None,-0.015,-0.015,-0.015,-0.015,-0.015],fmt=PCT,
     note='Diluted shares fell ~1.1%/yr FY23-FY26; new $5.4B authorization supports ~1.5%')
section(ws,15,'INCOME STATEMENT',10)
HL=lambda k,col: f'=Historicals!{col}{H[k]}'
mrow(16,'rev','Revenue',HL('rev','C'),HL('rev','D'),
     lambda c,i: f'={I["g26rev"]}' if i==0 else (f'={I["g27rev"]}' if i==1 else f'={prev(c)}16*(1+{c}6)'),bold=True,note='FY26E/27E = guidance')
mrow(17,'grw','  growth %','=B16/Historicals!B5-1','=C16/B16-1',lambda c,i: f'={c}16/{prev(c)}16-1',fmt=PCT)
mrow(18,'ebitda','Adj. EBITDA',HL('aebitda','C'),HL('aebitda','D'),lambda c,i: f'={c}16*{c}7',bold=True)
mrow(19,'ebm','  margin %','=B18/B16','=C18/C16',lambda c,i: f'={c}18/{c}16',fmt=PCT)
mrow(20,'da','Depreciation & amortization',HL('da','C'),HL('da','D'),lambda c,i: f'={c}16*{c}8')
mrow(21,'ebit','Adj. EBIT',f'=B18-B20',f'=C18-C20',lambda c,i: f'={c}18-{c}20')
mrow(22,'int','Interest expense, net',HL('int','C'),HL('int','D'),lambda c,i: f'={c}11')
mrow(23,'oth','Other expense, net',HL('other','C'),HL('other','D'),lambda c,i: f'={c}12')
mrow(24,'pbt','Adj. pre-tax income','=B21-B22-B23','=C21-C22-C23',lambda c,i: f'={c}21-{c}22-{c}23')
mrow(25,'tax','Taxes',None,None,lambda c,i: f'={c}24*{I["tax"]}')
mrow(26,'ni','Adj. net income',None,None,lambda c,i: f'={c}24-{c}25',bold=True)
mrow(27,'sh','Diluted shares (mm)',HL('dsh','C'),HL('dsh','D'),lambda c,i: '=Historicals!F19' if i==0 else f'={prev(c)}27*(1+{c}13)',fmt=NUM1,
     note='FY26E = 9M2026 diluted weighted average (402)')
mrow(28,'eps','Adj. EPS',None,None,lambda c,i: f'={c}26/{c}27',fmt=USD2,bold=True)
mrow(29,'epsg','  growth %',None,None,lambda c,i: None if i==0 else f'={c}28/{prev(c)}28-1',fmt=PCT)
mrow(30,'chk','  FY2026E check: model EPS vs. guidance $5.50',None,None,[f'=D28-{I["g26eps"]}',None,None,None,None,None],fmt=USD2,
     note='Small gap = model is consistent with management\'s guidance')
section(ws,32,'FREE CASH FLOW',10)
mrow(33,'capex','Capital expenditures',HL('capex','C'),HL('capex','D'),lambda c,i: f'={c}16*{c}9')
mrow(34,'dnwc','Increase in net working capital',None,None,lambda c,i: f'=({c}16-{prev(c)}16)*{c}10')
mrow(35,'nopat_tax','Taxes on EBIT (unlevered)',None,None,lambda c,i: f'={c}21*{I["tax"]}')
mrow(36,'ufcf','Unlevered FCF (EBITDA - tax on EBIT - capex - dNWC)',None,None,lambda c,i: f'={c}18-{c}35-{c}33-{c}34',bold=True,
     note='Cash available to ALL capital providers (debt + equity): what the DCF discounts')
mrow(37,'lfcf','Levered FCF (NI + D&A - capex - dNWC)',HL('fcf','C'),HL('fcf','D'),lambda c,i: f'={c}26+{c}20-{c}33-{c}34',bold=True,
     note='Cash available to equity. Compare with company FCF definition (CFO - capex)')
mrow(38,'fcfchk','  FY2026E check: model FCF vs. guidance $2,000',None,None,[f'=D37-{I["g26fcf"]}',None,None,None,None,None],
     note='Model below guidance: our NWC assumption is more conservative than management\'s plan')
mrow(39,'fcfps','Levered FCF per share',None,None,lambda c,i: f'={c}37/{c}27',fmt=USD2)
mrow(40,'fcfy','FCF yield at current price',None,None,lambda c,i: f'={c}39/{I["price"]}',fmt=PCT)
mrow(41,'pe','P/E at current price',None,None,lambda c,i: f'={I["price"]}/{c}28',fmt=MULT)
mrow(42,'evebitda','EV/EBITDA at current EV',None,None,lambda c,i: f'={I["ev"]}/{c}18',fmt=MULT)
ws.freeze_panes='B5'

# ================= DCF =================
ws=wb.create_sheet('DCF')
ws.column_dimensions['A'].width=46
for c in 'BCDEFGHI': ws.column_dimensions[c].width=12
ws.column_dimensions['J'].width=60
put(ws,'A1','Discounted Cash Flow Valuation ($mm, except per share)',TITLE)
header(ws,3,['','Q4-26E stub','FY2027E','FY2028E','FY2029E','FY2030E','FY2031E'])
mc=dict(zip('CDEFG','EFGHI'))  # DCF col -> Model col
label(ws,'A4','Unlevered free cash flow')
put(ws,'B4',f'={I["stub"]}',fmt=USD,note='Stub uses company FCF (levered) as a proxy: small and conservative')
for c in 'CDEFG': put(ws,f'{c}4',f'=Model!{mc[c]}36',fmt=USD)
label(ws,'A5','Cash-flow timing (mid-period, years from valuation date)')
put(ws,'B5',f'=(DATE(2026,12,31)-{I["valdate"]})/2/365.25',fmt=NUM2)
for i,c in enumerate('CDEFG'):
    put(ws,f'{c}5',f'=(DATE({2027+i},7,1)-{I["valdate"]})/365.25',fmt=NUM2)
label(ws,'A6','Discount factor')
for c in 'BCDEFG': put(ws,f'{c}6',f'=1/(1+{I["wacc"]})^{c}5',fmt='0.000')
label(ws,'A7','Present value of FCF',bold=True)
for c in 'BCDEFG': put(ws,f'{c}7',f'={c}4*{c}6',fmt=USD)
W=I['wacc']
blk=[
(9,'Sum of PV of FCF','=SUM(B7:G7)',USD,''),
(10,'Terminal year timing (end of FY2031, years)',f'=(DATE(2031,12,31)-{I["valdate"]})/365.25',NUM2,''),
(12,'A) PERPETUITY GROWTH METHOD',None,None,''),
(13,'Terminal value = FCF2031 x (1+g) / (WACC - g)',f'=G4*(1+{I["g"]})/({W}-{I["g"]})',USD,'Gordon growth formula'),
(14,'PV of terminal value',f'=B13/(1+{W})^B10',USD,''),
(15,'Enterprise value','=B9+B14',USD,''),
(16,'  % of EV from terminal value','=B14/B15',PCT,'Typically 60-80%: most value is beyond year 5'),
(17,'Less: net debt',f'={I["netdebt"]}',USD,''),
(18,'Equity value','=B15-B17',USD,''),
(19,'Value per share',f'=B18/({I["shares"]}+{I["dilut"]})',USD2,''),
(20,'  Upside / (downside) vs. current price',f'=B19/{I["price"]}-1',PCT,''),
(21,'  Implied exit EV/EBITDA multiple',f'=B13/Model!I18',MULT,'Sanity check: what multiple does this TV imply?'),
(23,'B) EXIT MULTIPLE METHOD',None,None,''),
(24,'Terminal value = FY2031E EBITDA x exit multiple',f'=Model!I18*{I["exitm"]}',USD,''),
(25,'PV of terminal value','=B24/(1+'+W+')^B10',USD,''),
(26,'Enterprise value','=B9+B25',USD,''),
(27,'  % of EV from terminal value','=B25/B26',PCT,''),
(28,'Equity value',f'=B26-{I["netdebt"]}',USD,''),
(29,'Value per share',f'=B28/({I["shares"]}+{I["dilut"]})',USD2,''),
(30,'  Upside / (downside) vs. current price',f'=B29/{I["price"]}-1',PCT,''),
(31,'  Implied perpetual growth rate',f'=(B24*{W}-G4)/(B24+G4)',PCT,'g = (TV x WACC - FCF) / (TV + FCF)'),
(33,'C) REVERSE DCF: WHAT IS PRICED IN?',None,None,''),
(34,'Current enterprise value',f'={I["ev"]}',USD,''),
(35,'Required PV of terminal value','=B34-B9',USD,''),
(36,'Implied terminal value (FY2031)','=B35*(1+'+W+')^B10',USD,''),
(37,'Implied exit EV/EBITDA on FY2031E',f'=B36/Model!I18',MULT,'The multiple the market must still pay in 2031 for today\'s price to earn exactly the WACC'),
(38,'Implied perpetual growth rate',f'=(B36*{W}-G4)/(B36+G4)',PCT,''),
]
for r,lab,f,fmt,n in blk:
    if f is None: section(ws,r,lab,10); continue
    label(ws,f'A{r}',lab,bold=lab.startswith('Value per share') or lab.startswith('Enterprise'))
    put(ws,f'B{r}',f,fmt=fmt); label(ws,f'C{r}',n,ital=True)
# Sensitivity tables: closed-form so they stay live
def sens(top,title,colvals,colfmt,cell_formula):
    section(ws,top,title,10)
    label(ws,f'A{top+1}','WACC (rows) vs. column variable',ital=True)
    for j,v in enumerate(colvals): put(ws,f'{L(2+j)}{top+1}',v,fmt=colfmt,font=Font(name=F,bold=True,color='0000FF',size=10))
    for i,w in enumerate([0.08,0.085,0.09,0.095,0.10,0.105,0.11]):
        rr=top+2+i
        put(ws,f'A{rr}',w,fmt=PCT,font=Font(name=F,bold=True,color='0000FF',size=10))
        for j in range(len(colvals)):
            cl=L(2+j)
            put(ws,f'{cl}{rr}',cell_formula(f'$A{rr}',f'{cl}${top+1}'),fmt=USD2)
pvsum=lambda w: f'SUMPRODUCT($B$4:$G$4,1/(1+{w})^$B$5:$G$5)'
sh=f'({I["shares"]}+{I["dilut"]})'
sens(41,'SENSITIVITY: value per share, perpetuity method (WACC x terminal growth)',[0.02,0.025,0.03,0.035,0.04],PCT,
     lambda w,g: f'=({pvsum(w)}+$G$4*(1+{g})/({w}-{g})/(1+{w})^$B$10-{I["netdebt"]})/{sh}')
sens(51,'SENSITIVITY: value per share, exit-multiple method (WACC x exit EV/EBITDA)',[14,18,22,26,30],MULT,
     lambda w,m: f'=({pvsum(w)}+Model!$I$18*{m}/(1+{w})^$B$10-{I["netdebt"]})/{sh}')
label(ws,'A60','Cross-check: B9 equals the SUMPRODUCT used in the tables at the base WACC ->',ital=True)
put(ws,'H60',f'=B9-{pvsum(I["wacc"])}',fmt=NUM2)
label(ws,'I60','(should be 0)',ital=True)

# ================= Comps =================
ws=wb.create_sheet('Comps')
widths=[8,30,12,10,11,11,11,11,9,9,11,11,10,10,10,10,10,9,9,9,9,9]
for i,w in enumerate(widths,1): ws.column_dimensions[L(i)].width=w
put(ws,'A1','Trading Comparables (LTM, GAAP, $mm except per share)',TITLE)
label(ws,'A2','Prices: Nov 6, 2026 close (stockanalysis.com). Fundamentals: SEC XBRL, latest 10-K + 10-Q; LTM = last fiscal year + current YTD - prior YTD. EBITDA = GAAP operating income + D&A (consistent across all names, so no company "adjustments").',ital=True)
cols=['Ticker','Company','LTM period end','Price','Shares (mm)','Class A price','Class A sh (mm)','Market cap','Cash','Debt','Noncontrol. int.','Enterprise value',
      'LTM revenue','LTM EBIT','LTM D&A','LTM EBITDA','LTM dil. EPS','EV / Revenue','EV / EBITDA','P / E','EBITDA margin']
header(ws,4,cols)
peers=[
('TDG','TransDigm Group','2026-06-27',1197.01,55.276525,None,None,2773,32621+139,9,10006,4608,420,32.97,
 'Debt = LongTermDebtAndCapitalLeaseObligations 32,621 + current 139. Sharecount: 10-Q cover Jul 31, 2026'),
('HEI','HEICO Corp (HEI + HEI.A)','2026-07-31',332.65,55.241647,250.28,84.515758,240.959,2537.660+3.513,77.243,5176.756,1244.555,217.592,6.00,
 'Two share classes; cover-page counts as of Aug 25, 2026 (10-Q filed 2026-08-27). P/E uses HEI (common) price'),
('ATI','ATI Inc.','2026-09-27',175.79,135.822697,None,None,442.3,1807.8+49.7,126.7,5013.4,826.3,177.0,4.00,
 'Debt = long-term 1,807.8 + current 49.7. Shares: 10-Q cover Oct 26, 2026'),
('CRS','Carpenter Technology','2026-09-30',278.31,49.540733,None,None,159.5,691.0,0,3188.7,745.1,147.7,11.34,
 'Fiscal year ends June 30. Shares: 10-Q cover Oct 16, 2026'),
('WWD','Woodward','2026-06-30',317.31,59.045820,None,None,474.851,617.730+131.779+592.426,0,4191.991,
 '=689.848+47.431-2.980',120.195,8.99,
 'No operating-income tag in XBRL: EBIT proxy = LTM pre-tax income 689.8 + interest exp 47.4 - interest income 3.0. Debt incl. short-term borrowings 592.4'),
]
r=5
for p in peers:
    t,n,per,px,shs,pa,sa,cash,debt,nci,rev,ebit,da,eps,src=p
    put(ws,f'A{r}',t,font=BOLD); label(ws,f'B{r}',n); put(ws,f'C{r}',per)
    put(ws,f'D{r}',px,fmt=USD2); put(ws,f'E{r}',shs,fmt=NUM1)
    if pa: put(ws,f'F{r}',pa,fmt=USD2); put(ws,f'G{r}',sa,fmt=NUM1)
    put(ws,f'H{r}',f'=D{r}*E{r}+N(F{r})*N(G{r})',fmt=USD)
    put(ws,f'I{r}',cash,fmt=USD); put(ws,f'J{r}',f'={debt}' if isinstance(debt,str) else debt,fmt=USD); put(ws,f'K{r}',nci,fmt=USD)
    put(ws,f'L{r}',f'=H{r}-I{r}+J{r}+K{r}',fmt=USD)
    put(ws,f'M{r}',rev,fmt=USD); put(ws,f'N{r}',ebit,fmt=USD); put(ws,f'O{r}',da,fmt=USD)
    put(ws,f'P{r}',f'=N{r}+O{r}',fmt=USD); put(ws,f'Q{r}',eps,fmt=USD2)
    put(ws,f'R{r}',f'=L{r}/M{r}',fmt=MULT); put(ws,f'S{r}',f'=L{r}/P{r}',fmt=MULT); put(ws,f'T{r}',f'=D{r}/Q{r}',fmt=MULT); put(ws,f'U{r}',f'=P{r}/M{r}',fmt=PCT)
    ws.cell(row=r,column=2).comment=Comment(src,'Analyst')
    r+=1
last=r-1
for lab,fn in [('Median','MEDIAN'),('Mean','AVERAGE')]:
    label(ws,f'B{r}',lab,bold=True)
    for c in 'RSTU': put(ws,f'{c}{r}',f'={fn}({c}5:{c}{last})',fmt=PCT if c=='U' else MULT,bold=True)
    for c in 'ABCDEFGHIJKLMNOPQRSTU': ws[f'{c}{r}'].border=TOP
    r+=1
med=r-2
r+=1
put(ws,f'A{r}','HWM',font=BOLD); label(ws,f'B{r}','Howmet Aerospace (at market)'); put(ws,f'C{r}','2026-09-30')
put(ws,f'D{r}',f'={I["price"]}',fmt=USD2); put(ws,f'E{r}',f'={I["shares"]}+{I["dilut"]}',fmt=NUM1)
put(ws,f'H{r}',f'=D{r}*E{r}',fmt=USD); put(ws,f'I{r}',f'={I["cash"]}',fmt=USD); put(ws,f'J{r}',f'={I["debt"]}',fmt=USD); put(ws,f'K{r}',0,fmt=USD)
put(ws,f'L{r}',f'=H{r}-I{r}+J{r}+K{r}',fmt=USD)
put(ws,f'M{r}','=Historicals!G5',fmt=USD); put(ws,f'N{r}','=Historicals!G11',fmt=USD); put(ws,f'O{r}','=Historicals!G9',fmt=USD)
put(ws,f'P{r}',f'=N{r}+O{r}',fmt=USD); put(ws,f'Q{r}','=Historicals!G20',fmt=USD2)
put(ws,f'R{r}',f'=L{r}/M{r}',fmt=MULT); put(ws,f'S{r}',f'=L{r}/P{r}',fmt=MULT); put(ws,f'T{r}',f'=D{r}/Q{r}',fmt=MULT); put(ws,f'U{r}',f'=P{r}/M{r}',fmt=PCT)
hw=r
r+=1
label(ws,f'B{r}','HWM premium / (discount) to peer median',ital=True)
for c in 'RST': put(ws,f'{c}{r}',f'={c}{hw}/{c}{med}-1',fmt=PCT)
r+=2
section(ws,r,'IMPLIED HWM VALUE FROM PEER MEDIAN MULTIPLES',21); r+=1
imp0=r
label(ws,f'B{r}','EV/EBITDA: peer median x HWM LTM EBITDA'); put(ws,f'H{r}',f'=S{med}*P{hw}',fmt=USD); label(ws,f'I{r}','implied EV',ital=True)
put(ws,f'L{r}',f'=(H{r}-{I["netdebt"]})/E{hw}',fmt=USD2); label(ws,f'M{r}','per share',ital=True); r+=1
label(ws,f'B{r}','P/E: peer median x HWM LTM diluted EPS'); put(ws,f'L{r}',f'=T{med}*Q{hw}',fmt=USD2); label(ws,f'M{r}','per share',ital=True); r+=1
label(ws,f'B{r}','Note: HWM LTM GAAP EBITDA includes a ~$91mm net restructuring credit (mostly the gain on the disk-forging plant sale) and FY2025 charges of $84mm, which roughly offset. Company Adj. EBITDA LTM = 9M26 $2,436 + Q4-25 $653 = $3,089.',ital=True)
COMPS_EVE=f'Comps!$L${imp0}'; COMPS_PE=f'Comps!$L${imp0+1}'
ws.freeze_panes='C5'

# ================= Summary =================
ws=wb.create_sheet('Summary',1)
ws.column_dimensions['A'].width=44
for c in 'BCDEF': ws.column_dimensions[c].width=14
ws.column_dimensions['G'].width=60
put(ws,'A1','HWM: Valuation Summary & Price Target',TITLE)
label(ws,'A2','All figures live-linked. Change yellow cells on Inputs/Model to re-run.',ital=True)
header(ws,4,['Method','Value / share','Weight','Weighted','vs. price','','Comment'])
meth=[('DCF: perpetuity growth','=DCF!B19',I['w1'],'Most conservative: assumes steady state after 2031'),
      ('DCF: exit multiple','=DCF!B29',I['w2'],'Depends heavily on the exit multiple you assume'),
      ('Comps: EV/EBITDA (peer median)',f'={COMPS_EVE}',I['w3'],'Peers growing slower than HWM: arguably understates'),
      ('Comps: P/E (peer median)',f'={COMPS_PE}',I['w4'],'GAAP LTM EPS basis')]
for i,(m,f,w,cmt) in enumerate(meth):
    r=5+i; label(ws,f'A{r}',m); put(ws,f'B{r}',f,fmt=USD2); put(ws,f'C{r}',f'={w}',fmt=PCT)
    put(ws,f'D{r}',f'=B{r}*C{r}',fmt=USD2); put(ws,f'E{r}',f'=B{r}/{I["price"]}-1',fmt=PCT); label(ws,f'G{r}',cmt,ital=True)
label(ws,'A9','Blended fair value today',bold=True); put(ws,'C9','=SUM(C5:C8)',fmt=PCT); put(ws,'D9','=SUM(D5:D8)',fmt=USD2,bold=True)
put(ws,'E9',f'=D9/{I["price"]}-1',fmt=PCT)
label(ws,'G9','Weights must sum to 100%',ital=True)
label(ws,'A11','Current price (Nov 6, 2026)'); put(ws,'B11',f'={I["price"]}',fmt=USD2)
label(ws,'A12','Cost of equity'); put(ws,'B12',f'={I["ke"]}',fmt=PCT)
label(ws,'A13','Forward 12-month dividends per share'); put(ws,'B13',0.56,fmt=USD2); label(ws,'G13','$0.14 quarterly dividend x 4 (Q3 2026 release)',ital=True)
label(ws,'A14','12-month price target (Nov 2027)',bold=True); put(ws,'B14','=D9*(1+B12)-B13',fmt=USD2,bold=True,fill=YEL)
label(ws,'G14','Fair value rolled forward one year at the cost of equity, less dividends received',ital=True)
label(ws,'A15','Expected 12-month total return',bold=True); put(ws,'B15','=(B14+B13)/B11-1',fmt=PCT,bold=True)
label(ws,'A16','Recommendation',bold=True)
put(ws,'B16',f'=IF(B15>{I["thresh"]},"LONG",IF(B15<-{I["thresh"]},"SHORT","NEUTRAL"))',bold=True)
label(ws,'G16','Rule from Inputs: >+15% LONG, <-15% SHORT, otherwise NEUTRAL',ital=True)
section(ws,18,'KEY MARKET-IMPLIED METRICS',7)
km=[('EV / FY2026E Adj. EBITDA (guidance)',f'={I["ev"]}/{I["g26ebitda"]}',MULT),
    ('P/E on FY2026E Adj. EPS (guidance)',f'={I["price"]}/{I["g26eps"]}',MULT),
    ('FCF yield on FY2026E FCF (guidance)',f'={I["g26fcf"]}/{I["mcap"]}',PCT),
    ('P/E on FY2027E Adj. EPS (model)','=Model!E41',MULT),
    ('Reverse DCF: implied FY2031 exit EV/EBITDA','=DCF!B37',MULT),
    ('WACC','='+I['wacc'],PCT)]
for i,(lab,f,fmt) in enumerate(km):
    label(ws,f'A{19+i}',lab); put(ws,f'B{19+i}',f,fmt=fmt)
label(ws,'A30','Robustness: blend EXCLUDING perpetuity DCF (equal weights on other 3)',bold=True)
put(ws,'B30','=AVERAGE(B6:B8)',fmt=USD2)
label(ws,'A31','  -> implied 12-month target'); put(ws,'B31','=B30*(1+B12)-B13',fmt=USD2)
label(ws,'A32','  -> expected total return'); put(ws,'B32','=(B31+B13)/B11-1',fmt=PCT)
label(ws,'A33','  -> recommendation under this alternative'); put(ws,'B33',f'=IF(B32>{I["thresh"]},"LONG",IF(B32<-{I["thresh"]},"SHORT","NEUTRAL"))')
label(ws,'G30','The call is sensitive to method weights: read this before publishing',ital=True)
section(ws,26,'BULL / BASE / BEAR (illustrative, per share)',7)
label(ws,'A27','Bear: DCF perpetuity at 10.5% WACC, 2.5% g'); put(ws,'B27','=DCF!C48',fmt=USD2)
label(ws,'A28','Base: blended fair value'); put(ws,'B28','=D9',fmt=USD2)
label(ws,'A29','Bull: exit multiple 26x (today\'s HWM multiple holds) at 8.5% WACC'); put(ws,'B29','=DCF!E54',fmt=USD2)
ch=BarChart(); ch.type='bar'; ch.title='Valuation football field ($/share)'; ch.style=10
ch.add_data(Reference(ws,min_col=2,min_row=4,max_row=8),titles_from_data=True)
ch.set_categories(Reference(ws,min_col=1,min_row=5,max_row=8)); ch.legend=None; ch.height=7; ch.width=16
ws.add_chart(ch,'A36')

for s in wb.worksheets:
    s.sheet_view.showGridLines=False
wb.save(OUT); print('saved',OUT)
