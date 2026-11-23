# Export Telegram messages for training

Use [Telegram Desktop](https://desktop.telegram.org/), the cross-platform desktop client. Telegram also distributes a separate native macOS app, so an installed app named Telegram does not by itself establish export support.

1. Open Telegram Desktop and sign in using Telegram's normal login process.
2. For selected chats, open a chat's **⋮ menu → Export chat history**. For an account export, use **Settings → Advanced → Export Telegram Data**.
3. Select **machine-readable JSON**, not HTML. Include the conversation's incoming and outgoing text. Turn media downloads off for this text-only workflow.
4. Select the intended chats and date range. Save outside the source checkout, in a private local folder.
5. Give the invoking Codex agent the resulting export path. The importer supports an individual chat JSON file or an account JSON containing `chats.list`.

No Telegram API ID, API hash, bot token, or third-party scraper is required. Do not copy Telegram authentication databases into the project.

After a new Desktop login, Telegram may require a 24-hour wait or confirmation of the export from an already logged-in device. The workflow cannot bypass this account-side restriction.

The agent must confirm the author's stable ID from the export/account information rather than infer authorship from display names or message frequency. The dataset target is 2,000 usable authored response episodes after curation, with 500 sufficient for an exploratory run. Raw export message count includes incoming messages and is not the training row count.

Sources: [Telegram's export announcement](https://telegram.org/blog/export-and-more), [Telegram's export instructions and new-login restriction](https://bugs.telegram.org/c/60/7).
