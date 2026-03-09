# Brief: Fight Night Event Promotion

**Branch:** scaviefae/event-promo
**Model:** sonnet
**Mode:** non-autonomous (human gates between passes)

## Goal

Get the word out about Agentic Smash Fight Night (Wednesday March 11, 7-10:30 PM PT, Frontier Tower SF) to the right audiences ASAP. Two days to drive attendance. Target: gaming × crypto × AI crossover crowd in SF during GDC/EthSF week.

**Event link:** https://luma.com/agentic-smash-fight-night

## Audience

- GDC attendees looking for afterparties/side events
- EthSF / crypto builders in SF that week
- SF Melee/Smash community
- AI/ML people who think "agents playing Smash" is cool
- General "what's happening in SF this week" crowd

## Pass 1: Channel Discovery

Search the web for every relevant channel. For each, record:

| Field | What |
|-------|------|
| Channel name | e.g. "GDC Events Calendar" |
| URL | Link to the listing/community |
| Type | Newsletter, event board, Discord, subreddit, Twitter account, Telegram, etc. |
| Audience | Who reads it, estimated size if findable |
| Submission process | How to get listed — form, DM, email, post directly |
| Deadline / timing | Is there a submission deadline? How fast do posts appear? |
| Relevance | High/Medium/Low for our audience |

**Search targets:**
- GDC official and unofficial event lists, afterparty aggregators, GDC fringe/satellite event boards
- EthSF event lists, Ethereum community calendars, crypto event aggregators for SF
- "This week in SF" / "SF events this week" general event lists and newsletters
- SF gaming communities (Melee locals, FGC groups, gaming Discord servers)
- SF tech/AI communities (meetup groups, newsletters, Twitter lists)
- Crypto Twitter accounts that amplify SF events
- Reddit: r/sanfrancisco, r/smashbros, r/ethereum, r/gamedev
- Luma discovery / trending (how to boost visibility there)

**Output:** `promo/channels.md` — the full table. Raw is fine, we'll prioritize in Pass 2.

**Done when:** At least 20 channels found across all categories. Pause for Mattie's review.

## Pass 2: Prioritize + Write Copy

Read Mattie's feedback in `promo/feedback.md` (if any). Then:

1. **Rank channels** by (audience size × relevance × timing feasibility). Drop anything that can't go live by Tuesday night.

2. **Write copy variants:**
   - **Tweet** (280 char) — punchy, shareable, tags relevant accounts
   - **Newsletter/event listing blurb** (100-150 words) — what, when, where, why it's cool, link
   - **Community post** (200-300 words) — more context, explains agents + Melee + betting, invites questions
   - **One-liner** — for DMs, group chats, quick shares

3. **Tailor copy per audience:**
   - Gaming crowd: lead with Melee, tournament format, "AI vs AI Smash Bros"
   - Crypto crowd: lead with onchain prediction markets, Monad, "bet on the match"
   - AI crowd: lead with autonomous agents, world models, "AI learned to play Melee"
   - General SF: lead with spectacle, free to watch, afterparty vibes

**Output:** `promo/plan.md` — ranked channel list with assigned copy variant per channel. Include the actual copy text.

**Done when:** Top 10 channels have specific copy written. Pause for Mattie's review.

## Pass 3: Execution Prep

Read Mattie's feedback. Then:

1. For each approved channel, prepare the exact submission:
   - Draft the post/submission in ready-to-paste format
   - Note any accounts that need to be logged into
   - Note any that Mattie needs to submit personally (e.g., her Twitter)
   - For event boards with forms, fill out a template of the form fields

2. Create `promo/tracker.md`:

| Channel | Copy variant | Status | Submitted by | Link | Notes |
|---------|-------------|--------|-------------|------|-------|
| ... | ... | draft / submitted / live / rejected | Mattie / auto | ... | ... |

**Output:** `promo/tracker.md` + all copy ready to go.

**Done when:** Every approved channel has a ready-to-submit draft. Mattie reviews, then submits or greenlights ScavieFae to submit where possible.

## Metrics

**Discovery quality (Pass 1):**
- Number of channels found (target: 20+)
- Category coverage (at least 3 channels per audience segment)
- No obvious gaps (Mattie gut check)

**Plan quality (Pass 2):**
- Top 10 channels identified with clear rationale
- Copy reads well, no LLM-speak, each variant feels native to its channel
- Timing is feasible (can go live by Tuesday night)

**Execution (Pass 3 + after):**
- Submissions made / planned
- Listings that go live / total submitted
- (Stretch) Luma registration spike after posts go live

## Notes

- Time is the constraint. We have ~36 hours for promo to have any effect. Prioritize channels with fast turnaround (post directly > submit for review).
- Don't spam. One post per community, well-written, native to the community's norms.
- Mattie's Twitter/social accounts are the primary megaphone. Copy for her to post is high priority.
- If a channel requires payment (promoted event listing), flag it — don't auto-purchase.
