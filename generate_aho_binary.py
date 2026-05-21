import csv

# 1. 保存するファイル名とヘッダー（列名）を指定
file_name = "aho_dataset_standard.csv"
headers = ["number", "output"]

# 2. ファイルを書き込みモードで開く（Excel対策で utf-8-sig を指定）
with open(file_name, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    
    # ヘッダーを書き込む
    writer.writeheader()
    
    # 3. ループを回しながら1行ずつ書き込む
    for i in range(1, 10000):
        # 3の倍数なら1、それ以外は0
        if i % 3 == 0:
            output = 1
        else:
            output = 0
            
        # 辞書形式で1行ずつ書き込み
        writer.writerow({
            "number": i,
            "output": output
        })

print("標準ライブラリでのCSV保存が完了しました！")