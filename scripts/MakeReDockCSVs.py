import pandas as pd
import os
import argparse

key = ['rec','tag']
delimiter=','
col_names=['tag', 'rmsd', 'rec']
header='infer'
num_unique=4260
def getPlottingDataFrame(path, col_names, header, delim, key, unique,exclusive_tag=None, rmsd_good = 2, max_N=9):  # unique is the number of unique receptor-ligand systems that should exist
    # Generates a dataframe used for the graph making functions
    # Each row of dataframe is a cumulation of statistics of all poses before and the current pose
    # Each row has the percentage of Receptor-Ligand Systems with less than 1, 2, and 3 RMSD for 'good1', 'good2', and 'good3' respectively
    initial_df = pd.read_csv(path, header=header, sep=delim, usecols=col_names)
    initial_df.columns = col_names
    initial_df['good'] = (initial_df['rmsd'] < rmsd_good)
    tags = initial_df[key[1]].unique()
    tags.sort()
    final_dataframe = pd.DataFrame(index=list(range(1, max_N+1)))
    for tag in tags:
        if exclusive_tag is not None and tag != exclusive_tag:
            print(tag)
            continue
        df_tagonly = initial_df[initial_df[key[1]] == tag]
        grouped_tagonly = df_tagonly.groupby(key[0])
        assert len(df_tagonly[key[0]].unique()) == unique, f"Doesn't have the right number of systems, should have {unique}, but has {len(df_tagonly[key[0]].unique())}"
        idx = grouped_tagonly.nth(0).index
        maxrange = 9
        rang = list(range(1, max_N+1))
        base = pd.DataFrame(None, index = idx, columns=['good'])
        combin_top_df = pd.DataFrame(None, index = rang, columns=['good']) #non_def_top -> combin_top_df
        top_bools = grouped_tagonly.nth(0)[['good']] #non_def_last -> top_bools
        combin_top_df.loc[1] = [top_bools['good'].mean()*100]
        for r in range(1, maxrange):
            cur_row = base.combine_first(grouped_tagonly.nth(r)[['good']]).fillna(False)
            top_bools = cur_row | top_bools
            combin_top_df.loc[r+1] = [top_bools['good'].mean()*100]
        final_dataframe[os.path.basename(file)] = combin_top_df['good']

    return final_dataframe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make CSVs for redock')
    #parser.add_argument('-i', '--input', nargs='+', help='Input file(s)', required=True)
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output', help='Output file', required=True)
    args = parser.parse_args()

    files = os.listdir(args.input)
    big_df = pd.DataFrame(index=list(range(1,10)))
    for file in files:
        if "redocking_" in file:
            if 'nocnn' not in file:
                use_cols=[0,2,7]
            else:
                use_cols=[0,2,4]
            plot_df = getPlottingDataFrame(os.path.join(args.input,file), col_names=col_names, header=header, delim=delimiter, key=key, unique=num_unique)
            big_df = big_df.join(plot_df)

    big_df.to_csv(args.output)


