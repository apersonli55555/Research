import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import load_workbook
OUT='/home/user/fall-detector/stock-pitches/2026-11-07_HWM/exhibits/paper/'
wb=load_workbook('/home/user/fall-detector/stock-pitches/2026-11-07_HWM/HWM_Model.xlsx',data_only=True)
SURF='#fcfcfb'; T1='#0b0b0b'; T2='#52514e'; GRID='#e4e3df'; HI='#2a78d6'; MUTE='#b9b8b2'; RED='#e34948'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.edgecolor':GRID,'axes.labelcolor':T2,
 'xtick.color':T2,'ytick.color':T2,'axes.facecolor':SURF,'figure.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False})
def finish(fig,ax,title,sub,src,name):
    fig.text(0.02,0.965,title,fontsize=14,fontweight='bold',color=T1,ha='left',va='top')
    fig.text(0.02,0.905,sub,fontsize=10.5,color=T2,ha='left',va='top')
    fig.text(0.02,0.02,src,fontsize=8,color=T2,ha='left',va='bottom')
    ax.grid(axis='y',color=GRID,lw=0.8); ax.set_axisbelow(True)
    fig.savefig(OUT+name,dpi=200); plt.close(fig)

# Figure 2: revenue growth by year (guided 2027 halves)
fig,ax=plt.subplots(figsize=(8,4.6)); fig.subplots_adjust(top=0.80,bottom=0.16,left=0.08,right=0.97)
yrs=['FY2024','FY2025','FY2026E\n(guidance)','FY2027E\n(guidance)']
gr=[7430/6640-1,8252/7430-1,10300/8252-1,11500/10300-1]
cols=[MUTE,MUTE,HI,RED]
b=ax.bar(yrs,[g*100 for g in gr],color=cols,width=0.55)
for r,g in zip(b,gr): ax.text(r.get_x()+r.get_width()/2,g*100+0.6,f'{g:.1%}',ha='center',color=T1,fontsize=11,fontweight='bold')
ax.set_ylabel('Revenue growth, %'); ax.set_ylim(0,30)
finish(fig,ax,'Figure 2: Management guides growth to roughly halve in 2027',
 '2026 includes the CAM and Brunner acquisitions; 2027 is management\'s first look (~$11.5B)',
 'Source: HWM FY2025 10-K; Q3 2026 earnings release (8-K Ex. 99.1, Oct 29, 2026). FY2026E = $10.3B guidance baseline.','fig2_growth.png')

# Figure 1: margin expansion
fig,ax=plt.subplots(figsize=(8,4.6)); fig.subplots_adjust(top=0.80,bottom=0.14,left=0.08,right=0.97)
lab=['FY2024','FY2025','9M2025','9M2026','FY2026E\n(guidance)']
m=[1914/7430,2416/8252,1763/6084,2436/7563,3330/10300]
b=ax.bar(lab,[x*100 for x in m],color=[MUTE,MUTE,MUTE,HI,HI],width=0.55)
for r,x in zip(b,m): ax.text(r.get_x()+r.get_width()/2,x*100+0.5,f'{x:.1%}',ha='center',color=T1,fontsize=11,fontweight='bold')
ax.set_ylabel('Adj. EBITDA margin, %'); ax.set_ylim(0,40)
finish(fig,ax,'Figure 1: The business is excellent: margins up ~650 bps in two years',
 'Adjusted EBITDA margin (company definition; stock-based compensation is NOT added back)',
 'Source: HWM Q4 2025 and Q3 2026 earnings releases (8-K Ex. 99.1); FY2026E = guidance baseline $3,330 / $10,300.','fig1_margin.png')

# Figure 5: drawdown since Aug 28
data=[('CRS',476.43,278.31),('HWM',264.85,205.81),('ATI',210.76,175.79),('ITA (A&D ETF)',77.61,68.98),('WWD',341.81,317.31),
      ('GE',342.58,323.24),('SPY (S&P 500)',769.35,735.29),('HEI',336.53,332.65),('TDG',1185.97,1197.01)]
fig,ax=plt.subplots(figsize=(8,5)); fig.subplots_adjust(top=0.82,bottom=0.17,left=0.20,right=0.95)
names=[d[0] for d in data][::-1]; rets=[(d[2]/d[1]-1)*100 for d in data][::-1]
b=ax.barh(names,rets,color=[HI if n=='HWM' else MUTE for n in names],height=0.6)
for r,v in zip(b,rets): ax.text(v-0.8 if v<0 else v+0.8,r.get_y()+r.get_height()/2,f'{v:+.1f}%',va='center',ha='right' if v<0 else 'left',color=T1,fontsize=10)
ax.axvline(0,color=T2,lw=1); ax.set_xlim(-50,8); ax.set_xlabel('Price change, Aug 28 to Nov 6, 2026 (%)')
ax.grid(axis='x',color=GRID,lw=0.8); ax.set_axisbelow(True)
fig.text(0.02,0.965,'Figure 5: The sell-off was concentrated in aerospace metals names',fontsize=14,fontweight='bold',color=T1,va='top')
fig.text(0.02,0.905,'HWM fell 22% alongside specialty-alloy makers CRS and ATI; TDG and HEI were roughly flat',fontsize=10.5,color=T2,va='top')
fig.text(0.02,0.02,'Source: daily closing prices, stockanalysis.com (retrieved Nov 7, 2026). Price return only.',fontsize=8,color=T2)
fig.savefig(OUT+'fig5_drawdown.png',dpi=200); plt.close(fig)

# Figure 3: growth-adjusted EV/EBITDA
c=wb['Comps']
rows=[(c[f'A{r}'].value,c[f'W{r}'].value) for r in range(5,10)]
rows+= [('HWM\ntrailing',c['W13'].value),('HWM\n2027E',c['W19'].value)]
rows=sorted(rows[:5],key=lambda x:x[1])+rows[5:]
fig,ax=plt.subplots(figsize=(8,4.8)); fig.subplots_adjust(top=0.80,bottom=0.22,left=0.08,right=0.97)
cols=[MUTE]*5+[HI,RED]
b=ax.bar([r[0] for r in rows],[r[1] for r in rows],color=cols,width=0.55)
for r,(n,v) in zip(b,rows): ax.text(r.get_x()+r.get_width()/2,v+0.04,f'{v:.2f}',ha='center',color=T1,fontsize=10.5,fontweight='bold')
med=c['W10'].value; ax.axhline(med,color=T2,lw=1,ls='--'); ax.text(-0.45,med+0.05,f'peer median {med:.2f}',ha='left',color=T2,fontsize=9)
ax.set_ylabel('EV/EBITDA ÷ revenue growth (pts)'); ax.set_ylim(0,2.4); ax.tick_params(axis='x',labelsize=9.5)
finish(fig,ax,'Figure 3: Cheap on trailing growth, expensive on next year\'s',
 'Growth-adjusted EV/EBITDA (lower = cheaper per point of growth). Trailing growth incl. acquisitions',
 'Source: SEC XBRL (latest 10-K/10-Q per company); prices Nov 6, 2026. Peers & HWM trailing: EV / LTM EBITDA ÷ YTD revenue growth.\nHWM 2027E: EV / FY2027E Adj. EBITDA ÷ 2027 guided growth (~11.7%).','fig3_growth_adj.png')

# Figure 6: football field + breakeven
s=wb['Summary']; d=wb['DCF']
meth=[('10-yr DCF\n(perpetuity)',s['B5'].value),('DCF\n(18x exit)',s['B6'].value),('Comps\nEV/EBITDA',s['B7'].value),('Comps\nP/E',s['B8'].value),('Blended\nfair value',s['D9'].value),('12-month\ntarget',s['B14'].value)]
fig,ax=plt.subplots(figsize=(8,4.8)); fig.subplots_adjust(top=0.80,bottom=0.16,left=0.08,right=0.97)
b=ax.bar([m[0] for m in meth],[m[1] for m in meth],color=[MUTE]*4+[HI,HI],width=0.55)
for r,(n,v) in zip(b,meth): ax.text(r.get_x()+r.get_width()/2,v+3,f'${v:.0f}',ha='center',color=T1,fontsize=11,fontweight='bold')
ax.axhline(205.81,color=RED,lw=1.5,ls='--'); ax.text(5.45,209,'price $205.81',ha='right',color=T1,fontsize=9.5)
ax.set_ylabel('$ per share'); ax.set_ylim(0,240)
finish(fig,ax,'Figure 6: All four methods land below the current price',
 'Value per share by method (WACC 10.45%, terminal growth 3%)',
 'Source: author\'s model (HWM_Model.xlsx), Nov 7, 2026.','fig6_valuation.png')

# Figure 4: 10-yr DCF vs WACC
w=[d.cell(row=87,column=k).value for k in range(2,8)]; v=[d.cell(row=88,column=k).value for k in range(2,8)]
fig,ax=plt.subplots(figsize=(8,4.6)); fig.subplots_adjust(top=0.80,bottom=0.15,left=0.09,right=0.97)
ax.plot([x*100 for x in w],v,color=HI,lw=2,marker='o',ms=7)
for x,y in zip(w,v): ax.text(x*100,y+9,f'${y:.0f}',ha='center',color=T1,fontsize=10)
ax.axhline(205.81,color=RED,lw=1.5,ls='--'); ax.text(10.4,211,'price $205.81',ha='right',color=T1,fontsize=9.5)
ax.axvline(6.64,color=T2,lw=1,ls=':'); ax.text(6.75,40,'break-even\nWACC ≈ 6.6%',color=T2,fontsize=9.5)
ax.set_xlabel('Discount rate (WACC), %'); ax.set_ylabel('$ per share'); ax.set_ylim(0,280)
finish(fig,ax,'Figure 4: To pay $206, you must discount HWM at ~6.6%',
 '10-year DCF value per share vs. discount rate; my CAPM WACC is 10.45%',
 'Source: author\'s model (HWM_Model.xlsx, DCF sheet section E). 10-year Treasury 4.84% (FRED DGS10, Nov 5, 2026).','fig4_breakeven.png')
print('done')
