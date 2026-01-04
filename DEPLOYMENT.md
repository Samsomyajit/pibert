# GitHub Pages Deployment Instructions

This repository is configured to automatically deploy the `index.html` file and all associated assets to GitHub Pages.

## Automatic Deployment

The deployment workflow (`.github/workflows/deploy-gh-pages.yml`) will automatically run when:
- Code is pushed to the `main` branch
- Manually triggered via the Actions tab

## Enabling GitHub Pages

To enable GitHub Pages for this repository, follow these steps:

1. Go to your repository on GitHub: https://github.com/Samsomyajit/pibert
2. Click on **Settings** (top navigation bar)
3. In the left sidebar, click on **Pages** (under "Code and automation")
4. Under **Source**, select **GitHub Actions** from the dropdown
5. Save the settings

## First Deployment

After enabling GitHub Pages with the GitHub Actions source:

1. The workflow will run automatically on the next push to `main`
2. Alternatively, you can manually trigger it:
   - Go to the **Actions** tab
   - Select the "Deploy to GitHub Pages" workflow
   - Click **Run workflow**
   - Select the `main` branch
   - Click **Run workflow**

## Accessing Your Site

Once deployed, your site will be available at:
```
https://samsomyajit.github.io/pibert/
```

## What Gets Deployed

The workflow deploys the entire repository root, including:
- `index.html` - Main website file
- All PNG images (PIBERT.png, PIBERTAbstract.png, eagle.png, cylinder.png, etc.)
- `outputs/` directory with sample images
- `LICENSE` file
- Any other assets referenced in the HTML

## Troubleshooting

If the deployment fails:
1. Check the Actions tab for error messages
2. Ensure GitHub Pages is enabled in repository settings
3. Verify that the workflow has the necessary permissions:
   - `contents: read`
   - `pages: write`
   - `id-token: write`

## Manual Deployment

To manually trigger a deployment:
1. Go to **Actions** > **Deploy to GitHub Pages**
2. Click **Run workflow**
3. Select the branch (usually `main`)
4. Click **Run workflow**

The deployment typically takes 1-2 minutes to complete.
