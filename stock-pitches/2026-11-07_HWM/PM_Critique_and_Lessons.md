# HWM Short: PM Critique and Teaching Notes

## Part 1: How a PM would tear this pitch apart

These are the questions I'd expect from a long/short PM, roughly in the order they'd ask them. Each comes with an honest answer.

### 1. "Your short depends on one valuation method. Defend the weights."
**This is the biggest weakness.** With equal weights the blend gives a $165 target (–19%) and a SHORT. Drop the perpetuity DCF and the target becomes $188 (–8%), which is NEUTRAL. That DCF implies Howmet is worth only 8.9x EBITDA in 2031, and no aerospace-component peer in the comp set trades anywhere near that today. A PM would say I picked a method that guarantees a short answer.
*Fix:* Either (a) extend the DCF to 10 years with growth fading gradually, so year 5 isn't treated as steady state, or (b) argue explicitly why multiples should compress. I'd rerun option (a) before increasing the position size.

### 2. "What's the catalyst? Why does this go down in the next 12 months?"
**Weak.** My catalysts are mostly things that "might disappoint." The company has beaten and raised every quarter this year, has $5.4B of buyback authorization, and has guided 2027 revenue up about 12%. A valuation short with no catalyst is how short sellers get hurt slowly.
*Better answer would need:* evidence that 2027 margin or EPS guidance will disappoint, that build rates are slowing, or that the 2026 spare-parts surge was pulled forward. I don't have that evidence yet.

### 3. "Why is the stock already down 22%? What does the market know?"
**I don't know, and I said so.** HWM fell from $264.85 (Aug 28) to $205.81 (Nov 6) even though Q3 results were strong. Before shorting, a PM would want to know whether this was a sector-wide de-rating, a rotation, or something company-specific. If it's sector-wide, I may be shorting near the bottom of a move that already happened.
*Homework:* Compare HWM's move with TDG, HEI, GE, and SPY over the same window.

### 4. "You're short a high-quality compounder. What's your risk/reward?"
Upside to the target is about –19% (my gain). The bull case in my model (26x exit multiple, 8.5% WACC) is $243, about +18% (my loss), and real squeezes can go further than that. That's roughly 1:1, which is mediocre for a short. Short losses are also uncapped while gains max out at 100%.
*Better framing:* Express it as a **pair trade**: short HWM, long a cheaper peer with similar end markets. On my comp sheet, CRS trades at 16.0x LTM EBITDA vs. HWM at 28.2x. A pair removes the "aerospace goes up" risk and isolates the valuation gap. **I have not done the work on CRS yet, so this is an idea, not a recommendation.**

### 5. "Your WACC is 10.45%. Isn't that too high for an A-/BBB+ rated company?"
That's a fair challenge. My beta of 1.20 matches published estimates (Yahoo 5Y monthly 1.19, Dividend Vision 1.21), but I didn't compute it myself. Many sell-side analysts use 8.5–9.5% for aerospace. At 9.0% WACC and 3% growth, the perpetuity DCF gives $112. That's still well below $206, so the conclusion holds. But the gap narrows a lot, and the exit-multiple DCF is barely affected by WACC.

### 6. "Your comps are flawed."
Yes, in four specific ways.
- **Growth isn't adjusted for.** HWM grew revenue 24% in 9M2026. Most peers are growing much slower. A growth-adjusted multiple (EV/EBITDA ÷ growth) would make HWM look less expensive.
- **Mismatched periods.** TDG, WWD, and HEI LTM periods end June/July 2026, while HWM's ends September 2026. TDG and WWD file 10-Ks in mid-November, so update the comps then.
- **WWD EBIT is a proxy** (pre-tax income + net interest), because Woodward doesn't tag operating income in XBRL.
- **HEI has two share classes.** I valued both, but the P/E uses the common (HEI) price, so it's overstated relative to HEI.A.

### 7. "Your FCF forecast is below management's. Why?"
My FY2026E levered FCF is $1,771M vs. guidance of $2,000M (–$229M). The difference is my working-capital assumption (12% of incremental sales). If I'm too conservative there, I'm too bearish on FCF every year. The 2025 working-capital build was 24.6% of incremental sales and 2024's was 15.6%, so my 12% isn't obviously too high. But it's a judgment call.

### 8. "What about the pension and leases?"
Net debt excludes the $493M noncurrent pension liability and $178M of lease liabilities. Including them would lower the value by about $1.70/share. That's not material to the call, but a careful analyst should mention it.

### PM verdict (my honest guess)
"Good work on the numbers. Everything ties to filings, and I like that you flagged the method sensitivity yourself. But this is a **watch-list short, not a position**. Come back when you have a catalyst, or restructure it as a pair."

---

## Part 2: Accounting and valuation concepts used in this pitch

### Reading the three statements
- **Income statement:** revenue minus costs equals profit, measured on an *accrual* basis. Revenue is recognized when earned, even if cash hasn't arrived yet. Howmet's 9M2026 operating income of $2,241M includes a **$91M one-time credit**, mostly the gain on selling the Savannah plant. A one-time item like that shouldn't be capitalized into a valuation, so analysts strip it out. Lesson: always read the "Restructuring and other" line.
- **Balance sheet:** a snapshot at a single date. Assets = Liabilities + Equity. For Howmet at Sep 30, 2026: $13,280M = $7,518M + $5,762M.
- **Cash flow statement:** reconciles net income to actual cash. Free cash flow = cash from operations – capex. Howmet's FY2025 FCF was $1,431M, **95% of net income**. High conversion like that suggests earnings aren't being inflated by accruals.

### GAAP vs. non-GAAP ("Adjusted")
Companies report "Adjusted EBITDA" by excluding items they call special. Always check the reconciliation in the earnings release. Howmet's Adj. EBITDA = Adj. operating income + D&A. It does **not** add back stock-based compensation, which keeps it more conservative than many tech companies' versions. I used company Adj. EBITDA for forecasts (it matches guidance) and GAAP for comps (so every company is measured the same way).

### LTM (last twelve months)
LTM = last fiscal year + current year-to-date – prior year-to-date. For HWM: $8,252 + $7,563 – $6,084 = **$9,731M** LTM revenue. This puts companies with different fiscal years on the same footing. In this pitch, CRS's year ends in June and TDG's in September.

### Working capital
Trade working capital = receivables + inventory – payables. It's cash tied up in running the business. A fast-growing manufacturer needs more of it, which is why FCF can lag profit during growth spurts. Howmet's trade working capital went from $1,783M to $2,260M in nine months.

### Enterprise value vs. equity value
- **Equity value (market cap)** = price × diluted shares = $82.3B.
- **Enterprise value** = equity + debt – cash (+ minority interest) = $86.2B. This is the price of the *whole business*, for both lenders and shareholders.
- Rule: match the numerator to the denominator. **EV goes with EBITDA**, a pre-interest profit that belongs to everyone. **Price goes with EPS**, a post-interest profit that belongs to shareholders.

### DCF in four steps
1. **Unlevered FCF** = EBITDA – taxes on EBIT – capex – increase in working capital. This is cash available to all capital providers.
2. **WACC** blends the cost of equity (CAPM: 4.84% risk-free + 1.20 beta × 5% premium = 10.8%) and the after-tax cost of debt (3.4%), weighted by market values (95% equity / 5% debt). The result is 10.45%.
3. **Terminal value** captures everything after 2031. There are two ways to estimate it:
   - Gordon growth: FCF × (1 + g) / (WACC – g)
   - Exit multiple: EBITDA × multiple
   Terminal value was 71–83% of total value here. That's normal, and it's why DCFs are so sensitive to their assumptions.
4. **Discount** each cash flow back to today. I used a mid-year convention, which assumes cash arrives evenly through the year. Then subtract net debt and divide by shares.

### Reverse DCF: the most useful tool for a short
Instead of asking "what is it worth?", you ask "what must I believe to justify today's price?" Here the answer is that HWM still trades at about 24x EBITDA in 2031, or grows FCF about 7.6% per year forever. Then you judge whether that's realistic. This keeps you honest, because you're debating the market's assumptions rather than your own.

### Sensitivity tables
Never present one number. The DCF sheet shows value across WACC (8–11%), terminal growth (2–4%), and exit multiples (14–30x). If the conclusion flips inside a reasonable range, conviction should be low. Here it partly does.

### Price target mechanics
Fair value today ($150) × (1 + cost of equity 10.8%) – dividends ($0.56) = **$165 12-month target**. The logic: if the stock were fairly priced, it should earn the cost of equity over the next year.

---

## Part 3: Process checklist for your next pitch
1. Pull the 10-K and latest 10-Q from EDGAR. Make sure the operating income and net income in your model tie exactly to the filing (the model has check rows for this).
2. Read the MD&A, segment note, acquisitions note, and risk factors. Most of the insight is in the notes, not the headline numbers.
3. Read the earnings release for guidance and the non-GAAP reconciliation.
4. Build projections anchored to guidance, then change one assumption at a time.
5. Run a reverse DCF *before* deciding long or short.
6. Write down your recommendation rule (for example ±15%) **before** you see the answer.
7. Write the bear case on your own thesis. If you can't, you don't understand it yet.
8. Log the pitch in the tracker on the publication date, and never edit an entry after publishing.
