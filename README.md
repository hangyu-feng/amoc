# amoc
Reproduction of Boers early warning signals

## Basic Git operations and workflows

1. Clone a repository
```bash
git clone <repo-url>
cd <repo-folder>
```

2. Create a branch
```bash
git branch <branch-name>      # create only
# or create and switch:
git switch -c <branch-name>
# (legacy) git checkout -b <branch-name>
```

3. Switch between branches
```bash
git switch <branch-name>
# (legacy) git checkout <branch-name>
```

4. Make changes, stage, commit, pull, and push
```bash
# stage changes:
git add <file>                # stage a specific file
git add --all                 # stage all changes (including deletions)
# or:
git add .                     # stage changes in current directory

git commit -m "Short message" # commit, put a somewhat meaningful message

# update your local branch from remote
git pull --rebase origin <branch-name>

# push your commits
git push origin <branch-name>
```

Using VS Code (Source Control panel)
- Open the Source Control panel (left sidebar icon or Ctrl+Shift+G).
- Stage files: click the + next to a changed file, or use the ... menu → Stage All.
- Commit: type a message in the message box and click the checkmark.
- Pull / Push: use the ... menu in the Source Control panel, the cloud icons, or the status bar sync button.
- Branches: click the branch name in the status bar or use the Command Palette (Ctrl+Shift+P) and run "Git: Create Branch" or "Git: Checkout to...".

Quick workflow summary:
- Clone the repo.
- Create and switch to a feature branch (CLI or VS Code).
- Make changes, stage (git add <file> or git add --all, or use Source Control), and commit.
- Pull to get remote updates, resolve conflicts if any.
- Push your branch and open a pull request.
