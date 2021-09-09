import os.path as osp


def getGitPath() -> osp:
    """
        a get function for the path to the git dir

        Returns
        -------
        git_dir: os.path - path to the git dir
    """
    current_dir = osp.dirname(osp.realpath(__file__))
    git_dir = osp.dirname(osp.dirname(current_dir))
    return git_dir
