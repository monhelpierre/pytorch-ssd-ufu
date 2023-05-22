==================================================
Testing with mobilenetV2_ssdlite
==================================================
100%|██████████████████████████████| 173/173 [14:39<00:00,  5.08s/it]
ID   Class            AP@[0.5]     AP@[0.5:0.95]
0    000000              0.865        0.542
1    000001              0.726        0.332
2    000003              0.699        0.387
3    000004              0.608        0.367
4    000007              0.802        0.422
5    000008              0.757        0.350
6    000009              0.860        0.491
7    000023              0.892        0.555
8    000025              0.897        0.570
9    000028              0.814        0.438
10   000035              0.898        0.529
11   000040              0.809        0.433
12   000042              0.777        0.508
13   000051              0.853        0.473
14   000052              0.749        0.333
15   000053              0.870        0.473
mAP@[0.5]: 0.805
mAP@[0.5:0.95]: 0.450

==================================================
Testing with mobilenetV3Large_ssdlite
==================================================
100%|██████████████████████████████| 173/173 [11:34<00:00,  4.02s/it]
ID   Class            AP@[0.5]     AP@[0.5:0.95]
0    000000              0.894        0.551
1    000001              0.552        0.266
2    000003              0.768        0.358
3    000004              0.769        0.467
4    000007              0.841        0.442
5    000008              0.850        0.381
6    000009              0.899        0.499
7    000023              0.883        0.506
8    000025              0.902        0.545
9    000028              0.752        0.357
10   000035              0.906        0.503
11   000040              0.837        0.415
12   000042              0.797        0.472
13   000051              0.841        0.458
14   000052              0.739        0.309
15   000053              0.884        0.431
mAP@[0.5]: 0.820
mAP@[0.5:0.95]: 0.435

==================================================
Testing with mobilenetV3Small_ssdlite
==================================================
100%|██████████████████████████████| 173/173 [06:29<00:00,  2.25s/it]
ID   Class            AP@[0.5]     AP@[0.5:0.95]
0    000000              0.658        0.380
1    000001              0.376        0.182
2    000003              0.236        0.113
3    000004              0.145        0.111
4    000007              0.430        0.199
5    000008              0.388        0.108
6    000009              0.507        0.276
7    000023              0.646        0.345
8    000025              0.722        0.370
9    000028              0.412        0.191
10   000035              0.599        0.314
11   000040              0.679        0.309
12   000042              0.540        0.307
13   000051              0.654        0.336
14   000052              0.456        0.164
15   000053              0.698        0.304
mAP@[0.5]: 0.509
mAP@[0.5:0.95]: 0.251






















Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Install the latest PowerShell for new features and improvements! https://aka.ms/PSWindows

PS C:\pytorch-ssd-ufu> git init
Initialized empty Git repository in C:/pytorch-ssd-ufu/.git/
PS C:\pytorch-ssd-ufu> git remote add origin git@github.com:monhelpierre/pythorch-ssd-ufu.git
PS C:\pytorch-ssd-ufu> git push origin master
error: src refspec master does not match any
error: failed to push some refs to 'github.com:monhelpierre/pythorch-ssd-ufu.git'
PS C:\pytorch-ssd-ufu> git push origin main
error: src refspec main does not match any
error: failed to push some refs to 'github.com:monhelpierre/pythorch-ssd-ufu.git'
PS C:\pytorch-ssd-ufu> git push origin master
error: src refspec master does not match any
error: failed to push some refs to 'github.com:monhelpierre/pythorch-ssd-ufu.git'
PS C:\pytorch-ssd-ufu> git branch -M main
PS C:\pytorch-ssd-ufu> git remote add origin git@github.com:monhelpierre/pythorch-ssd-ufu.git
error: remote origin already exists.
PS C:\pytorch-ssd-ufu> git push origin main
error: src refspec main does not match any
error: failed to push some refs to 'github.com:monhelpierre/pythorch-ssd-ufu.git'
PS C:\pytorch-ssd-ufu> git auth
git: 'auth' is not a git command. See 'git --help'.

The most similar commands are
        push
        status
PS C:\pytorch-ssd-ufu> git
usage: git [-v | --version] [-h | --help] [-C <path>] [-c <name>=<value>]
           [--exec-path[=<path>]] [--html-path] [--man-path] [--info-path]
           [-p | --paginate | -P | --no-pager] [--no-replace-objects] [--bare]
           [--git-dir=<path>] [--work-tree=<path>] [--namespace=<name>]
           [--config-env=<name>=<envvar>] <command> [<args>]

These are common Git commands used in various situations:

start a working area (see also: git help tutorial)
   clone     Clone a repository into a new directory
   init      Create an empty Git repository or reinitialize an existing one

work on the current change (see also: git help everyday)
   add       Add file contents to the index
   mv        Move or rename a file, a directory, or a symlink
   restore   Restore working tree files
   rm        Remove files from the working tree and from the index

examine the history and state (see also: git help revisions)
   bisect    Use binary search to find the commit that introduced a bug
   diff      Show changes between commits, commit and working tree, etc
   grep      Print lines matching a pattern
   log       Show commit logs
   show      Show various types of objects
   status    Show the working tree status

grow, mark and tweak your common history
   branch    List, create, or delete branches
   commit    Record changes to the repository
   merge     Join two or more development histories together
   rebase    Reapply commits on top of another base tip
   reset     Reset current HEAD to the specified state
   switch    Switch branches
   tag       Create, list, delete or verify a tag object signed with GPG

collaborate (see also: git help workflows)
   fetch     Download objects and refs from another repository
   pull      Fetch from and integrate with another repository or a local branch
   push      Update remote refs along with associated objects

'git help -a' and 'git help -g' list available subcommands and some
concept guides. See 'git help <command>' or 'git help <concept>'
to read about a specific subcommand or concept.
See 'git help git' for an overview of the system.
PS C:\pytorch-ssd-ufu> git clone https://github.com/monhelpierre/pythorch-ssd-ufu
Cloning into 'pythorch-ssd-ufu'...
info: please complete authentication in your browser...
remote: Enumerating objects: 3, done.
remote: Counting objects: 100% (3/3), done.
Receiving objects: 100% (3/3), 698 bytes | 349.00 KiB/s, done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
PS C:\pytorch-ssd-ufu> git add .
warning: in the working copy of 'LICENSE', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'configs/mobilenetV2_ssdlite.yaml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'configs/mobilenetV3Large_ssdlite.yaml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'configs/mobilenetV3Small_ssdlite.yaml', LF will be replaced by CRLF the next time Git touches it
warning: adding embedded git repository: pythorch-ssd-ufu
hint: You've added another git repository inside your current repository.
hint: Clones of the outer repository will not contain the contents of
hint: the embedded repository and will not know how to obtain it.
hint: If you meant to add a submodule, use:
hint:
hint:   git submodule add <url> pythorch-ssd-ufu
hint:
hint: If you added this path by mistake, you can remove it from the
hint: index with:
hint:
hint:   git rm --cached pythorch-ssd-ufu
hint:
hint: See "git help submodule" for more information.
warning: in the working copy of 'readme.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'report.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'requirements.txt', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'test.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'train.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/LICENSE', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/boxes.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/constants.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/data/dataloader.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/data/transforms.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/metrics.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/misc.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/models/__init__.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/models/backbones/__init__.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/models/backbones/mobilenetV2.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/models/backbones/mobilenetV3Large.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/models/backbones/mobilenetV3Small.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/models/layers.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/models/ops.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'utils/models/ssd.py', LF will be replaced by CRLF the next time Git touches it
PS C:\pytorch-ssd-ufu> git commit
Author identity unknown

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'monhe@LAB-PC.(none)')
PS C:\pytorch-ssd-ufu> git push origin main
error: src refspec main does not match any
error: failed to push some refs to 'github.com:monhelpierre/pythorch-ssd-ufu.git'
PS C:\pytorch-ssd-ufu> git commit
Author identity unknown

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'monhe@LAB-PC.(none)')
PS C:\pytorch-ssd-ufu> git commit -m
error: switch `m' requires a value
PS C:\pytorch-ssd-ufu> git commit -m "Initial upload"
Author identity unknown

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'monhe@LAB-PC.(none)')
PS C:\pytorch-ssd-ufu> git config --global user.email "monhemaudoonypierre@gmail.com"
PS C:\pytorch-ssd-ufu> git config --global user.name "monhelpierre"
PS C:\pytorch-ssd-ufu> git commit -m "Initial upload"
[main (root-commit) 9c68659] Initial upload
 155 files changed, 2005 insertions(+)
 create mode 100644 LICENSE
 create mode 100644 configs/mobilenetV2_ssdlite.yaml
 create mode 100644 configs/mobilenetV3Large_ssdlite.yaml
 create mode 100644 configs/mobilenetV3Small_ssdlite.yaml
 create mode 100644 dataset.py
 create mode 100644 labelimg.md
 create mode 160000 pythorch-ssd-ufu
 create mode 100644 readme.md
 create mode 100644 report.md
 create mode 100644 requirements.txt
 create mode 100644 results/mobilenetV2_ssdlite/best.pth
 create mode 100644 results/mobilenetV2_ssdlite/last.pth
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684498070.LAB-PC.11948.0
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684504662.LAB-PC.3352.0
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684516779.LAB-PC.1756.0
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684596832.LAB-PC.6764.0
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684596889.LAB-PC.14332.0
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684596981.LAB-PC.8368.0
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684597200.LAB-PC.6608.0
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684597248.LAB-PC.3772.0
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684599490.LAB-PC.13244.0
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684599703.LAB-PC.15524.0
 create mode 100644 results/mobilenetV2_ssdlite/train/events.out.tfevents.1684766241.LAB-PC.11252.0
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684498070.LAB-PC.11948.1
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684504662.LAB-PC.3352.1
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684516779.LAB-PC.1756.1
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684596832.LAB-PC.6764.1
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684596889.LAB-PC.14332.1
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684596981.LAB-PC.8368.1
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684597200.LAB-PC.6608.1
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684597248.LAB-PC.3772.1
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684599490.LAB-PC.13244.1
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684599703.LAB-PC.15524.1
 create mode 100644 results/mobilenetV2_ssdlite/val/events.out.tfevents.1684766241.LAB-PC.11252.1
 create mode 100644 results/mobilenetV3Large_ssdlite/train/events.out.tfevents.1684597848.nheltech-pc.24660.0
 create mode 100644 results/mobilenetV3Large_ssdlite/train/events.out.tfevents.1684598356.LAB-PC.12808.0
 create mode 100644 results/mobilenetV3Large_ssdlite/train/events.out.tfevents.1684599465.LAB-PC.16080.0
 create mode 100644 results/mobilenetV3Large_ssdlite/train/events.out.tfevents.1684599596.LAB-PC.8636.0
 create mode 100644 results/mobilenetV3Large_ssdlite/train/events.out.tfevents.1684766137.LAB-PC.13968.0
 create mode 100644 results/mobilenetV3Large_ssdlite/val/events.out.tfevents.1684597848.nheltech-pc.24660.1
 create mode 100644 results/mobilenetV3Large_ssdlite/val/events.out.tfevents.1684598356.LAB-PC.12808.1
 create mode 100644 results/mobilenetV3Large_ssdlite/val/events.out.tfevents.1684599465.LAB-PC.16080.1
 create mode 100644 results/mobilenetV3Large_ssdlite/val/events.out.tfevents.1684599596.LAB-PC.8636.1
 create mode 100644 results/mobilenetV3Large_ssdlite/val/events.out.tfevents.1684766137.LAB-PC.13968.1
 create mode 100644 results/mobilenetV3Small_ssdlite/best.pth
 create mode 100644 results/mobilenetV3Small_ssdlite/last.pth
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684597552.nheltech-pc.15264.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684597814.nheltech-pc.15492.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684598304.LAB-PC.13272.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684599444.LAB-PC.13456.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684625657.LAB-PC.15900.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684706061.nheltech-pc.15000.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684706360.LAB-PC.4916.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684706661.LAB-PC.8716.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684707075.LAB-PC.8128.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684707613.LAB-PC.15140.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684707673.LAB-PC.4968.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684708187.LAB-PC.14760.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684710513.LAB-PC.1952.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684754425.nheltech-pc.8760.0
 create mode 100644 results/mobilenetV3Small_ssdlite/train/events.out.tfevents.1684765953.LAB-PC.9324.0
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684597552.nheltech-pc.15264.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684597814.nheltech-pc.15492.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684598304.LAB-PC.13272.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684599444.LAB-PC.13456.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684625657.LAB-PC.15900.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684706061.nheltech-pc.15000.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684706360.LAB-PC.4916.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684706661.LAB-PC.8716.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684707075.LAB-PC.8128.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684707613.LAB-PC.15140.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684707673.LAB-PC.4968.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684708187.LAB-PC.14760.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684710513.LAB-PC.1952.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684754425.nheltech-pc.8760.1
 create mode 100644 results/mobilenetV3Small_ssdlite/val/events.out.tfevents.1684765953.LAB-PC.9324.1
 create mode 100644 test.py
 create mode 100644 test/images/download.jpg
 create mode 100644 test/images/images (1).jpg
 create mode 100644 test/images/images (2).jpg
 create mode 100644 test/images/images (3).jpg
 create mode 100644 test/images/images (4).jpg
 create mode 100644 test/images/images (5).jpg
 create mode 100644 test/images/images (6).jpg
 create mode 100644 test/images/images (7).jpg
 create mode 100644 test/images/images (8).jpg
 create mode 100644 test/images/images (9).jpg
 create mode 100644 test/images/images.jpg
 create mode 100644 train.py
 create mode 100644 utils/LICENSE
 create mode 100644 utils/__pycache__/boxes.cpython-310.pyc
 create mode 100644 utils/__pycache__/boxes.cpython-311.pyc
 create mode 100644 utils/__pycache__/boxes.cpython-39.pyc
 create mode 100644 utils/__pycache__/constants.cpython-310-nheltech-pc.pyc
 create mode 100644 utils/__pycache__/constants.cpython-310.pyc
 create mode 100644 utils/__pycache__/constants.cpython-311.pyc
 create mode 100644 utils/__pycache__/constants.cpython-39.pyc
 create mode 100644 utils/__pycache__/metrics.cpython-310-nheltech-pc.pyc
 create mode 100644 utils/__pycache__/metrics.cpython-310.pyc
 create mode 100644 utils/__pycache__/metrics.cpython-311.pyc
 create mode 100644 utils/__pycache__/metrics.cpython-39.pyc
 create mode 100644 utils/__pycache__/misc.cpython-310-nheltech-pc.pyc
 create mode 100644 utils/__pycache__/misc.cpython-310.pyc
 create mode 100644 utils/__pycache__/misc.cpython-311.pyc
 create mode 100644 utils/__pycache__/misc.cpython-39.pyc
 create mode 100644 utils/boxes.py
 create mode 100644 utils/constants.py
 create mode 100644 utils/data/__pycache__/dataloader.cpython-310.pyc
 create mode 100644 utils/data/__pycache__/dataloader.cpython-311.pyc
 create mode 100644 utils/data/__pycache__/dataloader.cpython-39.pyc
 create mode 100644 utils/data/__pycache__/transforms.cpython-310.pyc
 create mode 100644 utils/data/__pycache__/transforms.cpython-311.pyc
 create mode 100644 utils/data/__pycache__/transforms.cpython-39.pyc
 create mode 100644 utils/data/dataloader.py
 create mode 100644 utils/data/transforms.py
 create mode 100644 utils/metrics.py
 create mode 100644 utils/misc.py
 create mode 100644 utils/models/__init__.py
 create mode 100644 utils/models/__pycache__/__init__.cpython-310.pyc
 create mode 100644 utils/models/__pycache__/__init__.cpython-311.pyc
 create mode 100644 utils/models/__pycache__/__init__.cpython-39.pyc
 create mode 100644 utils/models/__pycache__/layers.cpython-310.pyc
 create mode 100644 utils/models/__pycache__/layers.cpython-311.pyc
 create mode 100644 utils/models/__pycache__/layers.cpython-39.pyc
 create mode 100644 utils/models/__pycache__/ops.cpython-310.pyc
 create mode 100644 utils/models/__pycache__/ops.cpython-311.pyc
 create mode 100644 utils/models/__pycache__/ops.cpython-39.pyc
 create mode 100644 utils/models/__pycache__/ssd.cpython-310.pyc
 create mode 100644 utils/models/__pycache__/ssd.cpython-311.pyc
 create mode 100644 utils/models/__pycache__/ssd.cpython-39.pyc
 create mode 100644 utils/models/backbones/__init__.py
 create mode 100644 utils/models/backbones/__pycache__/__init__.cpython-310-nheltech-pc.pyc
 create mode 100644 utils/models/backbones/__pycache__/__init__.cpython-310.pyc
 create mode 100644 utils/models/backbones/__pycache__/__init__.cpython-311.pyc
 create mode 100644 utils/models/backbones/__pycache__/__init__.cpython-39.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV1.cpython-310.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV2.cpython-310.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV2.cpython-311.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV2.cpython-39.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV3.cpython-310.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV3Large.cpython-310.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV3Large.cpython-311.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV3Large.cpython-39.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV3Small.cpython-310.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV3Small.cpython-311.pyc
 create mode 100644 utils/models/backbones/__pycache__/mobilenetV3Small.cpython-39.pyc
 create mode 100644 utils/models/backbones/__pycache__/vgg16.cpython-310.pyc
 create mode 100644 utils/models/backbones/__pycache__/vgg16.cpython-311.pyc
 create mode 100644 utils/models/backbones/__pycache__/vgg16.cpython-39.pyc
 create mode 100644 utils/models/backbones/mobilenetV2.py
 create mode 100644 utils/models/backbones/mobilenetV3Large.py
 create mode 100644 utils/models/backbones/mobilenetV3Small.py
 create mode 100644 utils/models/layers.py
 create mode 100644 utils/models/ops.py
 create mode 100644 utils/models/ssd.py
PS C:\pytorch-ssd-ufu> git push origin main
The authenticity of host 'github.com (20.201.28.151)' can't be established.
ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
This key is not known by any other names.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added 'github.com' (ED25519) to the list of known hosts.
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
PS C:\pytorch-ssd-ufu> ssh-keygen -R github.com
# Host github.com found: line 1
C:\Users\monhe/.ssh/known_hosts updated.
Original contents retained as C:\Users\monhe/.ssh/known_hosts.old
PS C:\pytorch-ssd-ufu> ssh-keygen -m PEM -t rsa -P "" -f afile
Too many arguments.
usage: ssh-keygen [-q] [-a rounds] [-b bits] [-C comment] [-f output_keyfile]
                  [-m format] [-N new_passphrase] [-O option]
                  [-t dsa | ecdsa | ecdsa-sk | ed25519 | ed25519-sk | rsa]
                  [-w provider] [-Z cipher]
       ssh-keygen -p [-a rounds] [-f keyfile] [-m format] [-N new_passphrase]
                   [-P old_passphrase] [-Z cipher]
       ssh-keygen -i [-f input_keyfile] [-m key_format]
       ssh-keygen -e [-f input_keyfile] [-m key_format]
       ssh-keygen -y [-f input_keyfile]
       ssh-keygen -c [-a rounds] [-C comment] [-f keyfile] [-P passphrase]
       ssh-keygen -l [-v] [-E fingerprint_hash] [-f input_keyfile]
       ssh-keygen -B [-f input_keyfile]
       ssh-keygen -D pkcs11
       ssh-keygen -F hostname [-lv] [-f known_hosts_file]
       ssh-keygen -H [-f known_hosts_file]
       ssh-keygen -K [-a rounds] [-w provider]
       ssh-keygen -R hostname [-f known_hosts_file]
       ssh-keygen -r hostname [-g] [-f input_keyfile]
       ssh-keygen -M generate [-O option] output_file
       ssh-keygen -M screen [-f input_file] [-O option] output_file
       ssh-keygen -I certificate_identity -s ca_key [-hU] [-D pkcs11_provider]
                  [-n principals] [-O option] [-V validity_interval]
                  [-z serial_number] file ...
       ssh-keygen -L [-f input_keyfile]
       ssh-keygen -A [-a rounds] [-f prefix_path]
       ssh-keygen -k -f krl_file [-u] [-s ca_public] [-z version_number]
                  file ...
       ssh-keygen -Q [-l] -f krl_file [file ...]
       ssh-keygen -Y find-principals -s signature_file -f allowed_signers_file
       ssh-keygen -Y check-novalidate -n namespace -s signature_file
       ssh-keygen -Y sign -f key_file -n namespace file ...
       ssh-keygen -Y verify -f allowed_signers_file -I signer_identity
                  -n namespace -s signature_file [-r revocation_file]
PS C:\pytorch-ssd-ufu> git push origin main
The authenticity of host 'github.com (20.201.28.151)' can't be established.
ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
This key is not known by any other names.
Are you sure you want to continue connecting (yes/no/[fingerprint])? fingerprint
Please type 'yes', 'no' or the fingerprint: yes
Warning: Permanently added 'github.com' (ED25519) to the list of known hosts.
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
PS C:\pytorch-ssd-ufu> git push origin master
error: src refspec master does not match any
error: failed to push some refs to 'github.com:monhelpierre/pythorch-ssd-ufu.git'
PS C:\pytorch-ssd-ufu> git config --global user.email "monhemaudoonypierre@gmail.com"
PS C:\pytorch-ssd-ufu> git connect
git: 'connect' is not a git command. See 'git --help'.
PS C:\pytorch-ssd-ufu> git config --global user.password ""
PS C:\pytorch-ssd-ufu> git push origin master
error: src refspec master does not match any
error: failed to push some refs to 'github.com:monhelpierre/pythorch-ssd-ufu.git'
PS C:\pytorch-ssd-ufu> git push origin main
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
PS C:\pytorch-ssd-ufu>  git config --global color.ui true
PS C:\pytorch-ssd-ufu>  ssh-keygen -t rsa -C "monhelmaudoonypierre@gmail.com"
Generating public/private rsa key pair.
Enter file in which to save the key (C:\Users\monhe/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in C:\Users\monhe/.ssh/id_rsa
Your public key has been saved in C:\Users\monhe/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:XsIn7ANnIwwcp6/gTYk+2egZ8kQMUQdGoqJyc6o4d7s monhelmaudoonypierre@gmail.com
The key's randomart image is:
+---[RSA 3072]----+
|oo=.o .          |
|.+ o +           |
|+   +            |
|oo . = o         |
|o B + = S o      |
|.= @ . B *       |
|. X +   +        |
|o*.+.    .       |
|oo+.Eo           |
+----[SHA256]-----+
PS C:\pytorch-ssd-ufu> ssh -T git@github.com
Enter passphrase for key 'C:\Users\monhe/.ssh/id_rsa':
Hi monhelpierre! You've successfully authenticated, but GitHub does not provide shell access.
PS C:\pytorch-ssd-ufu> git push origin main
Enter passphrase for key '/c/Users/monhe/.ssh/id_rsa':
To github.com:monhelpierre/pythorch-ssd-ufu.git
 ! [rejected]        main -> main (fetch first)
error: failed to push some refs to 'github.com:monhelpierre/pythorch-ssd-ufu.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
PS C:\pytorch-ssd-ufu> git remote update
Enter passphrase for key '/c/Users/monhe/.ssh/id_rsa':
remote: Enumerating objects: 3, done.
remote: Counting objects: 100% (3/3), done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), 678 bytes | 39.00 KiB/s, done.
From github.com:monhelpierre/pythorch-ssd-ufu
 * [new branch]      main       -> origin/main
PS C:\pytorch-ssd-ufu> git reset upstream/master
fatal: ambiguous argument 'upstream/master': unknown revision or path not in the working tree.
Use '--' to separate paths from revisions, like this:
'git <command> [<revision>...] -- [<file>...]'
PS C:\pytorch-ssd-ufu> git pull -r upstream master
fatal: 'upstream' does not appear to be a git repository
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
PS C:\pytorch-ssd-ufu> git add .
PS C:\pytorch-ssd-ufu> git commit -m "Initial upload"
On branch main
nothing to commit, working tree clean
PS C:\pytorch-ssd-ufu> git push origin main
Enter passphrase for key '/c/Users/monhe/.ssh/id_rsa':
To github.com:monhelpierre/pythorch-ssd-ufu.git
 ! [rejected]        main -> main (non-fast-forward)
error: failed to push some refs to 'github.com:monhelpierre/pythorch-ssd-ufu.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
PS C:\pytorch-ssd-ufu> git push --force origin main
Enter passphrase for key '/c/Users/monhe/.ssh/id_rsa':
Enumerating objects: 170, done.
Counting objects: 100% (170/170), done.
Delta compression using up to 8 threads
Compressing objects: 100% (168/168), done.
Writing objects: 100% (170/170), 33.08 MiB | 2.94 MiB/s, done.
Total 170 (delta 15), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (15/15), done.
To github.com:monhelpierre/pythorch-ssd-ufu.git
 + 95e9bb8...9c68659 main -> main (forced update)
PS C:\pytorch-ssd-ufu> git push --force origin main
Enter passphrase for key '/c/Users/monhe/.ssh/id_rsa':
Everything up-to-date
PS C:\pytorch-ssd-ufu>