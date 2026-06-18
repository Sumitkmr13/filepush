# SOW & Invoice Extraction App — User Guide

## What the app does

The app reads PDF contracts, SOWs, and licence documents from your SharePoint folder, extracts key fields using AI, and generates two Excel files:

- **contract_metrics.xlsx** — SOW / contract documents
- **license_metrics.xlsx** — Licence, subscription, and invoice documents

---

## Step 1 — Open the app

Open the app URL in your browser (provided by your IT/deployment team).

Example: `https://your-app-name.run.app`

---

## Step 2 — Sign in

Click **Sign in** and use your **Microsoft work account** (same account you use for Teams, Outlook, SharePoint).

> Only accounts approved by your IT team can access the app.

---

## Step 3 — Set your SharePoint folder

This is where you tell the app **which SharePoint folder to read PDFs from**.

### How to get the correct link from SharePoint

1. Open **SharePoint** in your browser.
2. Navigate to the folder containing the PDF files.
3. Click the **three dots (…)** next to the folder name.
4. Click **"Copy link"** (not the browser address bar URL — that format may not work).

   The link looks like:
   ```
   https://yourcompany.sharepoint.com/:f:/s/SiteName/AbCdEfGhIj...
   ```
   or
   ```
   https://yourcompany.sharepoint.com/:f:/r/sites/SiteName/...
   ```

   > ⚠️ **Do NOT copy from the browser address bar** while inside the folder. That URL format is not supported. Always use the **Copy link** option from the SharePoint menu.

5. Go back to the app.
6. Paste the copied link into the **"Paste SharePoint site OR full folder URL"** field.
7. Leave the **Drive path** and **Drive ID** fields empty.
8. Click **"Save SharePoint Context"**.

The app will validate your access. If successful, you will see a confirmation message. If you do not have access to that folder, the app will tell you.

---

## Step 4 — Start extraction

Once SharePoint context is saved:

1. Click **"Start Extraction"**.
2. The app will begin reading PDFs from your folder. Progress is shown in the **Processing Overview** panel at the bottom.
3. Do not close the browser tab while extraction is running; you can leave it open and it will update automatically.
4. When done, the status will change from **Running** to **Idle**.

> Files already processed in a previous run are **skipped automatically** (Smart Resume). Only new or changed files are re-processed.

---

## Step 5 — Download the Excel files

Once extraction finishes:

1. Go to the **Download Center** on the right side of the screen.
2. Click:
   - **"Download license_metrics.xlsx"** for licence/invoice/PO documents.
   - **"Download contract_metrics.xlsx"** for SOW/contract documents.

### Filtered download (optional)

You can download only rows within a date range:

1. Select the file from the dropdown.
2. Pick a **Start Date (from)** and **End Date (to)**.
3. Click **"Download Filtered Excel"**.

---

## Tips

| Situation | What to do |
|-----------|------------|
| "SharePoint context validation failed" | Check you used **Copy link** from SharePoint menu, not the browser bar |
| "Logged-in user does not have access" | Ask your SharePoint admin to grant you access to that folder |
| "No PDFs found at path" | The folder may be empty or the link points to a file, not a folder |
| Need to change folder | Paste a new link and click **Save SharePoint Context** again |
| Want to reprocess all files | Check **"Force reprocess"** and click **"Start Extraction"** |
| Session expired | Click **Logout** and sign in again |

---

## Logging out

Click the **Logout** button in the top control bar when you are done.

---

## Support

For access issues or errors not covered above, contact your IT/deployment team with a screenshot of the error message shown on screen.
