import os.path as osp


def getGitPath():
    current_dir = osp.dirname(osp.realpath(__file__))
    git_dir = osp.dirname(osp.dirname(current_dir))
    return git_dir
