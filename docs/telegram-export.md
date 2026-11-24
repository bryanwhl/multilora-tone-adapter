# Import Telegram exports

Use an existing Telegram Desktop HTML export directory or JSON export. The workflow reads local files; it needs no Telegram API ID, API hash, bot token, or live Telegram login. Keep all exports outside the source checkout.

## Existing HTML export

Pass one export directory, retaining its exported layout and message pages. Do not combine an HTML directory and JSON files in one import.

HTML does not consistently include stable account IDs or timezone-aware timestamps. Before labeling or training, confirm:

1. The exact exported sender name or names that represent your messages. Historical aliases can be supplied by repeating `--author-name`. A common name, high message count, or assumed “owner” does not prove authorship.
2. The IANA timezone used when the export was produced, such as `Europe/London`. This is not necessarily your current Mac timezone or where you now live.

Illustrative invocation:

```bash
uv run multilora inspect /path/to/telegram-html-export \
  --author-name "Your exported sender name" \
  --export-timezone Europe/London
uv run multilora prepare /path/to/telegram-html-export \
  --author-name "Your exported sender name" \
  --export-timezone Europe/London \
  --run telegram-html-pilot
```

Use the same confirmed identity and timezone for inspection and preparation. `--author-name` and JSON's `--author-id` are mutually exclusive. The example timezone is not an application default.

The importer interprets naive exported wall-clock times in the confirmed zone, then converts them to UTC. It quarantines timestamps that are ambiguous or nonexistent at daylight-saving transitions instead of guessing. Uncertain sender attribution, media-dependent content, and forwarded material are also quarantined. Review the aggregate report and source samples privately before the first labeling run.

HTML provenance uses export-local chat/message keys and source snapshot hashes. These are not Telegram API IDs. Keep the snapshot manifest with the run and do not assume a renamed folder or new export preserves source identity automatically. Source selection via `--chat-id` refers to the importer’s chat keys for that snapshot.

## JSON export

JSON is preferred when available because it provides stronger machine-readable identity and timestamp metadata. Existing supported HTML does not need to be re-exported solely to use this workflow.

To create a new JSON export, use [Telegram Desktop](https://desktop.telegram.org/), the cross-platform client. Telegram also distributes a separate native macOS client; an installed app named Telegram does not by itself establish export support.

1. Open Telegram Desktop and sign in through Telegram's normal login process.
2. Open a selected chat's **⋮ menu → Export chat history**, or use **Settings → Advanced → Export Telegram Data** for an account export.
3. Select **machine-readable JSON**, including incoming and outgoing text. Turn media downloads off for the text-only workflow.
4. Select chats and dates and save the export privately outside the source checkout.
5. Supply the JSON path and your stable exported user ID through `--author-id`. Do not infer that ID from display names.

```bash
uv run multilora inspect /path/to/telegram/result.json --author-id user123456
uv run multilora prepare /path/to/telegram/result.json \
  --author-id user123456 --run telegram-json-pilot
```

The importer accepts an individual chat JSON file or an account JSON containing `chats.list`. Do not copy Telegram authentication databases into this project.

A new Desktop login may require a 24-hour wait or confirmation from an already logged-in device before a new export. That account-side restriction is separate from importing an existing local export.

## Selection and usable data

Use repeatable `--chat-id` to restrict chat keys, `--since` for an inclusive lower bound, and `--until` for an exclusive upper bound. Calendar dates mean UTC midnight; ISO bounds require an explicit timezone. These filters operate on normalized UTC time for both formats.

Selection takes recent eligible episodes round-robin across chats up to `--limit`. Common secret patterns and media-bearing messages are screened before labeling; this is conservative and cannot detect every secret. Reports summarize exclusions without message bodies.

The target is 2,000 usable authored response episodes after curation; 500 permits an exploratory run. Raw message counts include incoming context and unusable records, so they are not training row counts.

Sources for creating new exports: [Telegram's export announcement](https://telegram.org/blog/export-and-more), [Telegram's export instructions and new-login restriction](https://bugs.telegram.org/c/60/7). Format-specific importer behavior is defined by this project's source and tests.
