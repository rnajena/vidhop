"""
MPT command line.
"""

import click
import sys
from vidhop.training.make_datasets import make_dataset
from vidhop.training.train_new_model import training
import os
# intended for faster cli due to lazy import of tensorflow, but not working yet
# import importlib
#
# class LazyLoader:
#     def __init__(self, lib_name):
#         self.lib_name = lib_name
#         self._mod = None
#
#     def __getattrib__(self, name):
#         if self._mod is None:
#             self._mod = importlib.import_module(self.lib_name)
#         return getattr(self._mod, name)


@click.group()
def entry_point():
    """
    VIDHOP is a virus host predicting tool. \b\n
    Its able to predict influenza A virus, rabies lyssavirus and rotavirus A.
    Furthermore the user can train its own models for other viruses and use them with VIDHOP.
    """
    pass


@entry_point.command(name="predict", short_help="predict the host of the viral sequence given")
@click.option('--input', '-i', required=True,
              help='either raw sequences or path to fasta file or directory with multiple files.')
@click.option('--virus', '-v', required=True, help='select virus species (influ, rabies, rota)')
@click.option('--outpath', '-o', help='path where to save the output')
@click.option('--n_hosts', '-n', default=int(0), help='show only -n most likely hosts')
@click.option('--thresh', '-t', default=float(0), help='show only hosts with higher likeliness then --thresh')
@click.option('--auto_filter', '-f', is_flag=True, help='automatically filters output to present most relevant host')
@click.version_option(version=0.9, prog_name="VIrus Deep learning HOst Predictor")
def cli(input, virus, outpath, n_hosts, thresh, auto_filter):
    '''
    predict the host of the viral sequence given

    \b
    Example:
    $ vidhop predict -i /home/user/fasta/influenza.fna -v influ
    \b
    present only hosts which reach a threshold of 0.2
    $ vidhop predict -i /home/user/fasta/influenza.fna -v influ -t 0.2
    \b
    if you want the output in a file
    $ vidhop predict -i /home/user/fasta/influenza.fna -v influ -o /home/user/vidhop_result.txt
    \b
    use multiple fasta-files in directory
    $ vidhop predict -i /home/user/fasta/ -v rabies
    \b
    use multiple fasta-files in directory and only present top 3 host predictions per sequence
    $ vidhop predict -i /home/user/fasta/ -v rabies -n_hosts
    \b
    Use your own trained models generated with vidhop training, specify path to the .model file you want to use.
    $ vidhop predict -v /home/user/out_training/model_best_acc_testname.model -i /home/user/fasta/

    '''

    assert virus in ["rota", "influ", "rabies"] or virus.endswith(
        ".model"), "not correct --virus parameter, use either rota, influ, rabies or path to .model file"
    assert thresh >= 0 and thresh <= 1, "error parameter --thresh: only thresholds between 0 and 1 allowed"
    assert n_hosts >= 0, "error parameter --n_hosts: only positive number of hosts allowed"

    from vidhop.vidhop_main import path_to_fastaFiles, start_analyses

    # prepare output path
    if outpath:
        if os.path.dirname(outpath) == outpath:
            outpath = os.path.join(outpath, "prediction.txt")
        outdir = os.path.dirname(outpath)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        sys.stdout = open(outpath, 'w')

    header_dict = path_to_fastaFiles(input)
    for key, value in header_dict.items():
        start_analyses(virus=virus, top_n_host=n_hosts, threshold=thresh, X_test_old=value, header=key,
                       auto_filter=auto_filter)


# if __name__ == '__main__':
entry_point.add_command(training)
entry_point.add_command(make_dataset)
entry_point()
