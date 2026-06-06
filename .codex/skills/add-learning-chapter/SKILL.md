---
name: add-learning-chapter
description: Propose and create numbered Markdown chapters in the repository's learning library by summarizing durable machine-learning and mathematics insights from the recent conversation and relevant project context. Use when the user asks to capture, record, document, or add recent learnings as a learning chapter. Always obtain explicit user confirmation of the proposed chapter before creating the file.
---

# Add Learning Chapter

Create concise, standalone chapters that preserve fundamental machine-learning
and mathematics understanding developed during the project.

## Workflow

1. Review the recent conversation for concepts the user has just understood,
   questioned, corrected, or connected.
2. Read relevant nearby project files only when needed to recover context or
   terminology.
3. Select durable lessons about machine learning or mathematics.
4. Exclude details that are mainly specific to a programming language,
   framework, library API, repository layout, or unusually narrow architecture.
5. Include those specific details only when the user explicitly asks to capture
   them, or when they are necessary to explain the underlying general concept.
6. Inspect `learnings/` and determine the next chapter number from existing
   three-digit filename prefixes.
7. Propose the chapter before editing anything. Show:
   - The proposed filename.
   - A short statement of scope.
   - The main ideas that will be included.
8. Ask the user to confirm the proposal.
9. Do not create or modify the chapter until explicit confirmation is received.
10. After confirmation, create the chapter and report its path.

If the recent discussion contains no durable ML or mathematics lesson, explain
that briefly and do not manufacture a chapter. Suggest the closest viable topic
and ask whether the user wants it captured.

## Filename Rules

Store chapters in `learnings/`.

Use:

```text
NNN-short-descriptive-title.md
```

Requirements:

- Use a three-digit, zero-padded chapter number such as `001`, `002`, or `010`.
- Increment the highest existing numeric prefix by one.
- Use lowercase kebab-case after the prefix.
- Never overwrite or reuse an existing chapter number.
- Keep the title focused on the general concept, not the immediate code change.

## Chapter Content

Write for future review without assuming the reader remembers the original
conversation.

Prefer this structure when it fits:

```markdown
# Clear Conceptual Title

## Core Idea

## How It Works

## Why It Matters

## Key Takeaway
```

Adapt the headings to the material rather than forcing empty sections.

Keep the chapter:

- Concept-first and technically precise.
- Concise enough to review quickly.
- Grounded with tensor shapes, equations, or small examples when useful.
- Explicit about distinctions that caused confusion.
- Free of conversation transcripts and unnecessary project history.
- Focused on transferable understanding rather than one-off implementation
  instructions.

Do not add advanced tangents that were not part of the learning being captured.
Do not turn a small insight into a broad textbook chapter.

## Confirmation Example

Before writing, respond in this form:

```text
Proposed chapter: learnings/002-cross-entropy-and-class-logits.md

Scope: Why a classifier produces one logit per possible class and how
cross-entropy uses those logits.

It will cover:
- The relationship between classes and output dimensions.
- The difference between logits and probabilities.
- Why the target is a class index.

Should I create this chapter?
```

Only create the file after the user clearly agrees.
