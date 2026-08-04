#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting Automated Deployment to GitHub..."
echo "--------------------------------------------------"

# 1. Check Git Repository status
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "❌ Error: Not a git repository. Please run 'git init' first."
    exit 1
fi

# 2. Stage all system files
echo "📦 Staging project files..."
git add .github/
git add .gitignore
git add best_config.json
git add auto_backtest_optimizer.py
git add scanner.py
git add train_gold_model.py
git add deploy.sh

# 3. Create Commit with custom message or default timestamp
COMMIT_MSG=$1
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="🤖 Update trading system and scanner pipeline ($(date +'%Y-%m-%d %H:%M:%S'))"
fi

echo "📝 Committing changes with message: '$COMMIT_MSG'..."
git commit -m "$COMMIT_MSG" || {
    echo "ℹ️ No changes detected to commit."
}

# 4. Detect default branch name (main or master)
BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")

# 5. Push to remote repository
echo "⬆️ Pushing changes to remote branch '$BRANCH'..."
git push origin "$BRANCH"

echo "--------------------------------------------------"
echo "✅ Deployment completed successfully!"
echo "📡 GitHub Actions will handle scanner and auto-retrain pipeline."
