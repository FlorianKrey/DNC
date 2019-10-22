"""Some util functions originally taken from PyHTK: https://ieeexplore.ieee.org/document/8683851"""
import argparse
import os
import logging
import sys
import copy

os.environ['STEPID'] = 'DNC'

class Abspath(argparse.Action):
    """Expand user- and relative-paths"""
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            if len(values) > 1:
                setattr(namespace, self.dest,
                        [os.path.abspath(os.path.expanduser(values[0]))] + values[1:])
            else:
                setattr(namespace, self.dest, [os.path.abspath(os.path.expanduser(values[0]))])
        else:
            setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

def print_error(message, exit_flag=True):
    """This function prints an error message and exit the program.

    :param message: The error message to print in the log.
    :type message: string
    :param exit: To exit after printing the message (default: True)
    :type exit: bool
    :raises: SystemExit

    """
    logging.getLogger(get_script_step_id()).error(message)
    if exit_flag:
        sys.exit(1)

def print_message(message):
    """This function prints a general information.

    :param message: The message to print in the log.
    :type message: str

    """
    logging.getLogger(get_script_step_id()).info(message)

def print_debug(message):
    """This function prints a debug message.

    :param message: The debug message to print in the log.
    :type message: str

    """
    logging.getLogger(get_script_step_id()).debug(message)

def set_script_step_id(stepid):
    """
    """
    if stepid is None:
        return
    stepid = get_file_name_base(stepid)
    set_env_var_value('STEPID', stepid)

def get_script_step_id():
    """
    """
    return get_env_var_value('STEPID')

def get_env_var_value(envvar):
    """
    """
    if envvar not in os.environ:
        print_error('Cannot load environment variable %s' % envvar)
    return os.environ.get(envvar)

def set_env_var_value(envvar, varval):
    """
    """
    os.environ[envvar] = varval

def get_abs_path(filepath):
    """Return the absolute path of filepath.

    :param filepath: The input file path to extend to absolute.
    :type filepath: string
    :returns: string -- The absolute path.

    """
    if filepath is None:
        return None
    return os.path.abspath(filepath)

def get_rel_path(filepath):
    """Return the relative path to filepath.

    :param filepath: The input file path to convert to relative.
    :type filepath: string
    :return: string -- The relative path.

    """
    return os.path.relpath(filepath)

def get_base_name(filepath):
    """
    """
    return os.path.basename(filepath)

def get_file_name_base(filepath):
    """
    """
    return os.path.splitext(get_base_name(filepath))[0]

def get_dir_name(filepath):
    """
    """
    return os.path.dirname(filepath)

def join_paths(path, *paths):
    """
    """
    return os.path.join(path, *paths)

def check_output_dir(filepath, overwrite=True):
    """Check whether the output path exists and is an output directory. If not exists then creates.

    :param filepath: The output directory path.
    :type filepath: string
    :param overwrite: Whether to overwrite in the target file (default: True).
    :type overwrite: bool
    :returns: string -- The directory path

    """
    if isinstance(filepath, list):
        for each_filepath in filepath:
            check_output_dir(each_filepath, overwrite)
        return
    if filepath is None or filepath == '':
        return
    if not check_exists(filepath):
        make_dirs(filepath)
    elif not overwrite:
        print_error('Output path %s exists, remove before re-runing' % filepath)
    if not check_is_dir(filepath):
        print_error('Output path %s is a file instead of a directory' % filepath)

def check_output_file(filepath, overwrite=True):
    """Check whether the output path points to a file and can create the directories if not exists.

    :param: filepath: The output file path.
    :type filepath: string
    :param overwrite: Whether to overwrite in the target file (default: True).
    :type overwrite: bool
    :returns: string -- The file path.

    """
    if isinstance(filepath, list):
        for each_filepath in filepath:
            check_output_file(each_filepath, overwrite)
        return
    if filepath is None or filepath == '':
        return
    if check_exists(filepath):
        if not check_is_file(filepath):
            print_error('Output path %s does not point to a file' % filepath)
        elif not overwrite:
            print_error('Output file path %s exists, remove before re-running' % filepath)
    else:
        dirpath = get_dir_name(filepath)
        check_output_dir(dirpath, overwrite)

def check_is_file(filepath):
    """Check whether the specified path points to a file.

    :param filepath: The file path to test.
    :type filepath: string
    :param has2exist: Whether the file path has to exist when return True.
    :type has2exist: bool
    :returns: bool -- Whether the filepath is a directory or not.

    """
    if not check_exists(filepath):
        return True
    elif os.path.isfile(filepath):
        return True
    else:
        return False

def cache_command(argvs, argtgt='', cmdtype=''):
    """
    """
    #print_message('Cache command')
    items = copy.copy(argvs)
    items[0] = get_rel_path(items[0])
    cmdline = ' '.join(items)
    cmdpath = join_paths('CMDs', argtgt, get_file_name_base(items[0]) + '.cmds')
    if cmdtype != '':
        cmdpath = join_paths('CMDs', argtgt, '%s.cmds' % cmdtype)
    cmdsep = '------------------------------------'
    write_one_text_file([cmdsep, cmdline, cmdsep], cmdpath, True)

def change_dir(filepath):
    """Change the current working directory.

    :param filepath: The directory to change to.
    :type filepath: string

    """
    if not check_is_dir(filepath, True):
        print_error('Directory %s does not exist' % filepath)
    print_message('Change working dir to %s' % filepath)
    os.chdir(filepath)

def check_exists(filepath):
    """
    """
    return os.path.exists(filepath)

def make_dirs(filepath):
    """Make the target directory.

    :param filepath: The directory to make.
    :type filepath: string

    """
    if isinstance(filepath, list):
        for eachpath in filepath:
            make_dirs(eachpath)
    else:
        filepaths = []
        if filepath.count('{') and filepath.count('}'):
            before = filepath.split('{')[0]
            middles = filepath.split('{')[-1].split('}')[0].split(',')
            after = filepath.split('}')[-1]
            for eachmid in middles:
                filepaths.append(join_paths(before, eachmid, after))
        else:
            filepaths = [filepath]
        for eachpath in filepaths:
            if eachpath == '' or eachpath is None:
                continue
            if check_exists(eachpath):
                if check_is_file(eachpath):
                    print_error('Target path %s is an existing file' % eachpath)
            else:
                print_debug('Creating dir %s' % eachpath)
                os.makedirs(eachpath)

def check_is_dir(filepath, has2exist=False):
    """Check whether the specified path points to a directory.

    :param filepath: The file path to test.
    :type filepath: string
    :param has2exist: Whether the file path has to exist when return True.
    :type has2exist: bool
    :returns: bool -- Whether the filepath is a directory or not.

    """
    if not check_exists(filepath):
        return False if has2exist else True
    elif os.path.isdir(filepath):
        return True
    else:
        return False

def write_one_text_file(lines, path, append=True):
    """Write to one text file and add new line symbol if needed

    :param lines: A list of lines to write out
    :type lines: list
    :param path: The output file path
    :type path: str
    :param append: Whether to append in the target file (default: True).
    :type append: bool

    """
    print_debug('Write: %s' % path)
    check_output_file(path, True)
    mode = 'w'
    if append:
        mode = 'a'
    if not isinstance(lines, list):
        lines = [lines]
    file = open(path, mode)
    for eachline in lines:
        curline = str(eachline)
        if not curline.endswith(os.linesep):
            curline += os.linesep
        file.write(curline)
    file.close()
