---
name: changelog-entry
description: Write a changelog entry in this project's house format. Use when the user asks to add, draft, or format a changelog/release note.
---

# Changelog entry

Follow [Keep a Changelog](https://keepachangelog.com) conventions, with this
project's house rules layered on top.

## Steps

1. Classify the change into exactly one section: **Added**, **Changed**,
   **Fixed**, **Deprecated**, **Removed**, or **Security**.
2. Write one line, imperative mood, present tense ("Add", not "Added"/"Adds").
3. End the line with the PR reference in parentheses, e.g. `(#1421)`.
4. Keep it under 100 characters. No trailing period.

## House rules (read `template.md` for the exact skeleton)

- Group entries under the unreleased heading `## [Unreleased]`.
- User-facing language only — describe the effect, not the implementation.
- One entry per behavioural change; split unrelated changes into separate lines.

When you need the precise file skeleton to paste into, read the bundled
`template.md` resource. Do not load it unless you are actually writing the file.
