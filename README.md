# Transformer Encoderの構造

- 数字の加工
	- output: [B, 8] 8は数字の桁数
- 単語埋め込み
	- output: [B, 8, 64] 64は単語ベクトルの長さ
- 位置エンコーディング
	- output: [B, 8, 64]
- Q, K, Vにする
	- w_q, w_q, w_v: [64, 64]
	- output: [B, 8, 64]
- マルチヘッドに分割
	- output: [B, 8, 4, 16] 4はヘッド数、16は一個一個のヘッドの大きさ
- 計算したいのは(8, 16)の部分なので、順番を入れ替える
	- output: [B, 4, 8, 16]
- attentionの算出(Q ・ K^T)
	- output: [B, 4, 8, 16] ・ [B, 4, 16, 8] = [B, 4, 8, 8]
- attentionの算出(Vをかける)
	- output: [B, 4, 8, 16]
- 順番をもとに戻す
	- output: [B, 8, 4, 16]
- ヘッド結合
	- output: [B, 8, 64]
- 残差接続
	- output: [B, 8, 64]
- 層正規化
	- output: [B, 8, 64]
- フラットにする
	- output: [B, 512]
- 線形分類層
	- output: [B, 1]
- シグモイド変換
	- output: [B, 1]

# Transformer Decoderの構造

- 数字の加工
	- output: [B, 8] 8は数字の桁数
- 単語埋め込み
	- output: [B, 8, 64] 64は単語ベクトルの長さ
- 位置エンコーディング
	- output: [B, 8, 64]
- Q, K, Vにする
	- w_q, w_q, w_v: [64, 64]
	- output: [B, 8, 64]
- マルチヘッドに分割
	- output: [B, 8, 4, 16] 4はヘッド数、16は一個一個のヘッドの大きさ
- 計算したいのは(8, 16)の部分なので、順番を入れ替える
	- output: [B, 4, 8, 16]
- attentionの算出(Q ・ K^T)
	- output: [B, 4, 8, 16] ・ [B, 4, 16, 8] = [B, 4, 8, 8]
- Casual Mask
	- output: [B, 4, 8, 8]
- attentionの算出(Vをかける)
	- output: [B, 4, 8, 16]
- 順番をもとに戻す
	- output: [B, 8, 4, 16]
- ヘッド結合
	- output: [B, 8, 64]
- 残差接続
	- output: [B, 8, 64]
- 層正規化
	- output: [B, 8, 64]
- FFNの中間層へ
	- output: [B, 8, 128]
- FFNを抜ける
	- output: [B, 8, 64]
- 残差接続
	- output: [B, 8, 64]
- 層正規化
	- output: [B, 8, 64]
- 出力全結合層
	- output: [B, 8, V] Vは文字の種類数
- 予測対象のみ抽出
	- output: [B, V]
- 損失計算＆推論予測 ロジットが最大値となる語彙IDを選択
	- output: [B]

# Q, K, Vの計算方法

- Q、K、Vに分岐する直前の行列をAとする：[B, 64, d_model]
- w_q, w_k, w_v = [d_model, d_model]
- Q = A・w_q^T(K, Vについても同様)
- 分割(例えば、8ヘッドに分割)
- Q・K^T / √d_k = [B, 8, 8]
- softmaxをかける
- Vをかける

# 学習時と生成時におけるAttention計算の違い

- encoder内部のself-attention、およびcross-attentionに関して変化はない
- casual-attentionについては、学習時には未来の情報をマスクして並列して未来予測を行うが、生成時には逐次処理をする必要がある。
- なお、生成時にはKVキャッシュを利用し、新たな文字を生成した際、その分だけattentionの計算を行うことで、計算量を減らしている。

## 自分が実装中に理解に苦労した点

- 3次元以上のテンソルの計算
- nn.moduleの仕様。initに何を書いてforwardに何を書くのか
- nn.moduleにおいて、インスタンスを作るときは__init__に渡す引数を渡して、そのインスタンスで何かを実行するときにはforwardに渡す引数を渡すことに気が付くのに時間がかかった。
- 次のpythonの仕様がわからなかった。重みを定義するとき、self.w_q = nn.Linear(d_model, d_k, bias=False)と書くと、内部的には(d_k, d_model)って形の行列ができる。しかし、実際に計算するとき、self.w_q(x)と書くと、x×(w_qの転置)の計算が行われるため、結局、x×(d_model, d_k)の計算が行われていることになる。