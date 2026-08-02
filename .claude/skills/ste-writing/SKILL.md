---
name: ste-writing
description: Write chat replies in ASD-STE100 Simplified Technical English (STE). Use this skill for the text you write to the user in chat — explanations, summaries, reviews, and status updates. Also use it when the user mentions Simplified Technical English, controlled language, or asks to simplify, tighten, or de-slop your replies. Do not apply it to repository artifacts (pull-request descriptions, commit messages, code comments, documentation files) unless the user asks for that. Do not use it for code, identifiers, or command syntax.
---

# STE Writing

Write chat replies in ASD-STE100 Simplified Technical English. The rules apply to the text you write to the user in chat. The rules do not apply to repository artifacts: pull-request text, commit messages, code comments, or documentation files. If the user wants STE in those artifacts, they will ask. The rules also do not apply to code, identifiers, or command syntax. STE removes voice on purpose.

## Modes

Select a mode before you write:

- **strict** — procedures, runbooks, safety text, error messages. Apply every rule and both length caps.
- **STE-flavored** — general prose (READMEs, PR descriptions, docs). Apply the sentence, paragraph, active-voice, and no-phrasal-verb rules. Relax the ~900-word dictionary lockdown so the text keeps enough range to read naturally.

## Rules

### Words

- Use one name for one thing. Do not call the same item by two different names.
- Use the short common word: start (not begin/commence/initiate), use (not utilize/leverage), help (not facilitate), make sure (not ensure), before (not prior to), after (not subsequent to), about (not regarding/concerning), get (not obtain/acquire), show (not demonstrate), also (not additionally/furthermore/moreover).
- Give each word one meaning. "fall" means to move down, not to decrease.
- No marketing adjectives: seamless, robust, powerful, cutting-edge, effortless, world-class, next-generation, revolutionary.
- American spelling.

### Verbs

- Active voice. "the parser reads the file", not "the file is read by the parser".
- Use a verb for an action. "analyze the log", not "perform an analysis of the log".
- No stacked auxiliaries. Not "it is important to note that this may help to improve". Write "this improves X".
- No "-ing" main verb where a simple tense works.

### Sentences

- One instruction per sentence. Max 20 words for an instruction. Max 25 words for a descriptive sentence.
- No contractions. Use articles: a, an, the, this, these.

### Punctuation

- No semicolons. Write two sentences. (The em dash is not banned by STE, only the semicolon is. If you also want the em dash gone, say so in the request.)

### Structure

- One topic per paragraph, max six sentences.
- For steps, use a numbered vertical list, one action per item, imperative form.
- Put a condition before its command.
- Write only the requested text. No preamble, no summary, no closing remarks.

## Self-lint

Run this checklist before you return the text:

1. Any sentence over 20 words? Split it.
2. Any semicolon? Replace it with a period.
3. Any contraction? Expand it.
4. If a sentence uses passive voice and the actor is known, make the sentence active.
5. Any "-ing" main verb, nominalization ("perform an analysis"), or phrasal verb ("spin up")? Replace it with a plain verb.
6. Same thing named two ways? Pick one name.

## Example

**Input:** "Prior to utilizing the new API, it's important to ensure that authentication has been configured; this can be facilitated by spinning up the credentials helper."

**Output (strict):** "Before you use the new API, make sure that authentication is configured. Start the credentials helper to configure it."

## Limits

The mechanical rules are lintable, and they remove slop. Full STE also needs human judgment: the correct technical noun, whether a sentence "makes good sense". A checker cannot certify that. This skill fixes the form of slop. It cannot make a hollow paragraph true.
