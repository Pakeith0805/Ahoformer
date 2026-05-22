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
    for i in range(1, 101):
        # 3の倍数か3がつく数字なら1、それ以外は0
        is_multiple_of_3 = (i % 3 == 0)
        contains_3 = ('3' in str(i))
        if is_multiple_of_3 or contains_3:
            output = 1
        else:
            output = 0
            
        # 辞書形式で1行ずつ書き込み
        writer.writerow({
            "number": i,
            "output": output
        })

print("標準ライブラリでのCSV保存が完了しました！")