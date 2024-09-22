#!/usr/bin/env python2
import os
import yaml
import numpy as np

# 处理数据统计和生成LaTeX表格
# 计算和保存数据统计信息，并将这些信息以LaTeX表格的形式呈现，以便在文档中使用


# 计算给定数据向量（data_vec）的统计信息，包括均值（mean）、中位数（median）、标准差（std）、最小值（min）、最大值（max）、均方根误差（rmse）和样本数量（num_samples）
# 如果数据向量为空，则所有统计值都设置为0
def compute_statistics(data_vec):
    stats = dict()
    if len(data_vec) > 0:
        stats['rmse'] = float(
            np.sqrt(np.dot(data_vec, data_vec) / len(data_vec)))
        stats['mean'] = float(np.mean(data_vec))
        stats['median'] = float(np.median(data_vec))
        stats['std'] = float(np.std(data_vec))
        stats['min'] = float(np.min(data_vec))
        stats['max'] = float(np.max(data_vec))
        stats['num_samples'] = int(len(data_vec))
    else:
        stats['rmse'] = 0
        stats['mean'] = 0
        stats['median'] = 0
        stats['std'] = 0
        stats['min'] = 0
        stats['max'] = 0
        stats['num_samples'] = 0

    return stats

# 如果指定的YAML文件（yaml_filename）存在，则加载现有的统计数据。
# 将新的统计数据（new_stats）添加到现有的统计数据中，以指定标签（label）进行标识。
# 将更新后的统计数据保存回YAML文件。
def update_and_save_stats(new_stats, label, yaml_filename):
    stats = dict()
    if os.path.exists(yaml_filename):
        stats = yaml.load(open(yaml_filename, 'r'), Loader=yaml.FullLoader)
    stats[label] = new_stats

    with open(yaml_filename, 'w') as outfile:
        outfile.write(yaml.dump(stats, default_flow_style=False))

    return

# 首先调用compute_statistics函数计算统计数据。
# 然后调用update_and_save_stats函数将新的统计数据保存到YAML文件中
def compute_and_save_statistics(data_vec, label, yaml_filename):
    new_stats = compute_statistics(data_vec)
    update_and_save_stats(new_stats, label, yaml_filename)

    return new_stats

# 生成一个LaTeX表格，用于将列表中的值写入表格中。
# 表格的标题行由列名（cols）组成，表格的行由行名（rows）和相应的值（list_values）组成。
# 表格中的值需要以LaTeX格式编写，以确保它们可以被直接复制到LaTeX文档中
def write_tex_table(list_values, rows, cols, outfn):
    '''
    write list_values[row_idx][col_idx] to a table that is ready to be pasted
    into latex source

    list_values is a list of row values

    The value should be string of desired format
    '''

    assert len(rows) >= 1
    assert len(cols) >= 1

    with open(outfn, 'w') as f:
        # write header
        f.write('      &      ')
        for col_i in cols[:-1]:
            f.write(col_i + ' & ')
        f.write(' ' + cols[-1]+'\n')

        # write each row
        for row_idx, row_i in enumerate(list_values):
            f.write(rows[row_idx] + ' &     ')
            row_values = list_values[row_idx]
            for col_idx in range(len(row_values) - 1):
                f.write(row_values[col_idx] + ' & ')
            f.write(' ' + row_values[-1]+' \n')
