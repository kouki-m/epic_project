import pandas as pd
import numpy as np
import argparse
import os

def trimmed_mean(df, trim_percent=20):
    """
    指定されたパーセンテージの上振れと下振れをカットして平均を取る関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        解析対象のデータフレーム
    trim_percent : float
        上下からカットするパーセンテージ (デフォルト: 10%)
    
    Returns:
    --------
    pandas.Series
        各カラムのトリム平均
    """
    # NumPyのtrimmedmean関数を使用
    # 各カラムに対して上下からtrim_percentずつカットして平均を計算
    trimmed_means = {}
    
    for column in df.columns:
        if column == 'Run':
            continue
        
        data = df[column].values
        lower_bound = np.percentile(data, trim_percent)
        upper_bound = np.percentile(data, 100 - trim_percent)
        
        # トリミングされたデータを取得
        trimmed_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        # トリミングされたデータの平均を計算
        trimmed_means[column] = np.mean(trimmed_data)
    
    return pd.Series(trimmed_means)

def analyze_csv(file_path, trim_percent=20):
    """
    CSVファイルを読み込み、トリム平均を計算する関数
    
    Parameters:
    -----------
    file_path : str
        解析対象のCSVファイルパス
    trim_percent : float
        上下からカットするパーセンテージ
    """
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)
    
    # トリム平均を計算
    trimmed_means = trimmed_mean(df, trim_percent)
    
    # 結果を表示
    print(f"Trimmed Mean ({trim_percent}% cut from top and bottom):")
    for column, mean in trimmed_means.items():
        print(f"{column}: {mean:.6f} ms")
    
    # 全体の平均実行時間も表示
    print(f"\nOverall trimmed execution time: {trimmed_means['Total']:.6f} ms")
    
    # # 各ステップが全体に占める割合を計算して表示
    # print("\nPercentage of total execution time:")
    # for column, mean in trimmed_means.items():
    #     if column != 'Total':
    #         percentage = (mean / trimmed_means['Total']) * 100
    #         print(f"{column}: {percentage:.2f}%")
    
    # 結果をCSVファイルに保存
    output_file = os.path.splitext(file_path)[0] + f"_analysis_{trim_percent}pct.csv"
    trimmed_means.to_frame('Trimmed Mean (ms)').to_csv(output_file)
    print(f"\nResults saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Epic times CSV with trimmed mean')
    parser.add_argument('file', help='Path to the CSV file')
    parser.add_argument('--trim', type=float, default=20.0, 
                        help='Percentage to trim from top and bottom (default: 10%%)')
    
    args = parser.parse_args()
    
    analyze_csv(args.file, args.trim)

if __name__ == "__main__":
    main()
