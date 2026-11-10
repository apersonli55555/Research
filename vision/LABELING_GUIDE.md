# FallOrNot labelling guide

Written after the first agreement study (run 8: two annotators, Cohen's kappa 0.50). Most disagreements were borderline events that one annotator called a fall and the other did not. This guide settles those cases. It does not change the definition; it makes it specific.

## Definition

A **fall** is an event in which a **person** inadvertently comes to rest on the ground, floor or another lower level (WHO).

Label the 4-second clip, in which the accident is around the 2-second mark. Three questions decide it:

1. **Is it a person?** Animals, toys, robots and objects are never falls.
2. **Was it unintended?** Lying down, sitting down, diving or jumping on purpose is not a fall, even if it ends badly later.
3. **Does the body come to rest lower?** The torso or hips must end up on the ground, floor, water or another lower surface. Hands or one knee touching down briefly is not enough.

All three must be yes, and it must be visible in the clip.

## Borderline cases

| Situation | Label | Why |
|---|---|---|
| Stumbles or slips but stays on their feet | not a fall | body does not come to rest lower |
| Catches themselves on hands or one knee and gets straight back up | not a fall | brief contact, no rest |
| Ends up sitting or lying on the ground, even for a moment | **fall** | body at rest on a lower level |
| Knocked down by a person, animal, ball or object and ends up on the ground | **fall** | the cause does not matter, the outcome does |
| Falls off a bike, skateboard, horse, chair, ladder or trampoline onto the ground | **fall** | lower level reached unintentionally |
| Falls into water unintentionally (pool, lake, off a dock) | **fall** | water is a lower level |
| Jumps or dives on purpose and lands badly but as intended (in the water, on the mat) | not a fall | intended descent |
| Jumps on purpose, then loses control and ends up on the ground | **fall** | the landing on the ground was not intended |
| Vehicle crash where no person visibly ends up on the ground | not a fall | no person at rest lower |
| Something falls on a person, who stays upright | not a fall | |
| Sits or flops down hard on purpose | not a fall | intended |
| A chair, hammock or swing breaks and the person drops to the ground | **fall** | |
| The fall starts but the clip ends before the person is down | fall only if it is clear they are going down; otherwise unsure | |
| The person who falls is in the background | **fall** | any person in the clip counts |
| Several people, one falls | **fall** | |
| A baby or toddler topples over while learning to walk or play | **fall** | a person, unintended |
| Too dark, blurry, far away or cut to tell | unsure | excluded from results |

## How to label

- Decide from the clip only, not from the video title.
- Do not discuss clips with another annotator while labelling.
- If you hesitate for more than two viewings, choose **unsure**.

## Settling disagreements (adjudication)

When two annotators disagree, both read this guide, watch the clip again together and agree on one label using the table. The tool shows only the disagreed clips and does not show who said what. If they still cannot agree, the label is **unsure**. Adjudicated labels are stored separately (`labels_adjudicated.csv`); the original labels of both annotators are kept unchanged.
