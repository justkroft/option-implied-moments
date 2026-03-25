[![English](https://img.shields.io/badge/English-blue)](./README.md)
[![日本語](https://img.shields.io/badge/日本語-red)](./README.ja.md)

# 株式オプション価格から推定されるモーメント
本リポジトリは、株式オプションデータからオプション・インプライド（すなわちリスク中立的）高次モーメントを計算するためのコードを含んでいる。ボラティリティ、歪度、尖度といった株式リターンの特性は、リスク中立測度 $\mathbb{Q}$ のもとで、アウト・オブ・ザ・マネーの株式オプション契約に基づいて算出される。

[[1]](#1) の手法を用いて、リスク中立的なボラティリティ、歪度、および尖度を計算する。同論文では、アウト・オブ・ザ・マネーのコールおよびプットの集合から、リスク中立リターン分布のモーメントの計算は以下のように定義される。

$$
Var^{\mathbb{Q}} = \frac{e^{r\tau}V_{i,t} - \mu^{2}}{\tau},
$$
$$
Skew^{\mathbb{Q}} = \frac{e^{r\tau}W - 3\mu e^{r\tau}V + 2\mu^{3}}{[e^{r\tau}V-\mu^{2}]^{3/2}},
$$
$$
Kurt^{\mathbb{Q}} = \frac{e^{r\tau}X - 4\mu e^{r\tau}W + 6e^{r\tau}\mu^{2}V - 3\mu^{4}}{[e^{r\tau}V-\mu^{2}]^{2}} - 3
$$

上記の数式では

$$
\mu = e^{r\tau} - 1 - \frac{e^{r\tau}}{2}V - \frac{e^{r\tau}}{b}W - \frac{e^{r\tau}}{24}X
$$

$V$、$W$、および $X$ は、測度 $\mathbb{Q}$ の下で、時点 $t$ から $t+\tau$ までの期間における株式リターンの対数の2乗、3乗、および4乗の期待値をそれぞれ表す。また、$r$ は同期間に対応する連続複利の無リスク金利である。
これらのパラメータを推定するために、 [[2]](#2) における台形則（trapezoidal rule）に基づく実装に従って、コールおよびプット・オプションの離散的なストライク価格を用いる。実装の詳細については、同論文の付録Bを参照してください。

## プログラム実装の詳細
本ライブラリのアーキテクチャは、3つの層から構成される。

> 注意：本ライブラリは、WRDS などのデータプロバイダから取得されるインプライド・ボラティリティ・サーフェス・データを用いる場合に最も良好に動作する。

### C層

数値計算はC言語で実装されている。
- 正規累積分布関数（CDF） — Abramowitz & Stegun による有理関数近似（§26.2.17）を用いて近似される。
- ブラック・ショールズ価格付け — アウト・オブ・ザ・マネーのオプション価格は、インプライド・ボラティリティから標準的な閉形式解を用いて計算され、累積正規分布には A&S 近似が適用される。
- コールおよびプットはストライク価格でソートされ、それぞれ価格付けされた後、[[2]](#2) に従い台形則によって個別に積分される。

### Cython層

Cython層は、NumPyとC拡張の橋渡しとして機能する。Cythonの関数は、CSR（Compressed Sparse Row）形式で配置されたフラットなNumPy配列を入力として受け取る。ストライク価格、インプライド・ボラティリティ、およびフラグはすべてのグループにわたって連結され、`indptr` 配列が各グループの境界を示す。
その後、各グループは OpenMP を用いて並列処理される。

### Python層

Python層はデータ前処理のみを担当する。具体的には、アウト・オブ・ザ・マネーのオプションのフィルタリング、コール・プットのフラグの整数の符号化、株および時間区間ごとのデータのソートとグループ化を行って、Cython層が要求するCSR形式を構築する。出力は Polars のDataFrameとして再構築される。

# 使用方法

セットアップ手順および使用例については、[英語版README](./README.md)を参照してください。

# 参考文献
<a id="1">[1]</a>
Bakshi, G., Kapadia, N., & Madan, D. (2003). Stock return characteristics, skew laws, and the differential pricing of individual equity options. *The Review of Financial Studies, 16*(1), 101-143.

<a id="2">[2]</a>
Bali, Turan G. and Hu, Jianfeng and Murray, Scott, Option Implied Volatility, Skewness, and Kurtosis and the Cross-Section of Expected Stock Returns (January 1, 2019). Georgetown McDonough School of Business Research Paper, Available at SSRN: https://ssrn.com/abstract=2322945 or http://dx.doi.org/10.2139/ssrn.2322945
