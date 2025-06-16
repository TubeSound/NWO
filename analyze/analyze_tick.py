import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
from dateutil import tz

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 

def read_data(symbol):
    filepath = f"../../MarketData/Axiory/{symbol}/Tick/{symbol}_TICK_2024_07-2025_05.csv" 
    df = pd.read_csv(filepath)
    df['jst'] = pd.to_datetime(df['jst'], format='ISO8601')
    return df

def make_features(df, crash_periods):
    # 初期化
    df['label'] = 0

    # 時間でラベルを1にする
    for start, end in crash_periods:
        start_ts = pd.to_datetime(start).tz_localize(JST)
        end_ts = pd.to_datetime(end).tz_localize(JST)
        df.loc[(df['jst'] >= start_ts) & (df['jst'] <= end_ts), 'label'] = 1

    # チェック：ラベル数の確認
    print(df['label'].value_counts())

    # === 3. 特徴量作成 ===
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    df['mid_diff'] = df['mid_price'].diff()
    df['spread'] = df['ask'] - df['bid']
    df['time_diff'] = df['jst'].diff().dt.total_seconds().fillna(0)
    df['abs_diff'] = df['mid_diff'].abs()
    df['volatility'] = df['abs_diff'] / df['time_diff'].replace(0, 1)
    df['volatility_ma'] = df['volatility'].rolling(5, min_periods=1).mean()

    # === 4. 欠損除去 ===
    df_feature = df.dropna(subset=['mid_diff', 'spread', 'time_diff', 'abs_diff', 'volatility', 'volatility_ma', 'label'])

    # === 5. 学習用データの準備 ===
    features = ['mid_diff', 'spread', 'time_diff', 'abs_diff', 'volatility_ma']
    X = df_feature[features]
    y = df_feature['label']
    
    return df_feature, X, y

def training(X, y):
    # 一部だけで学習（高速化用に10万件に制限）
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    print('y_train:', y_train.value_counts())

    # === 6. モデル学習 ===
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def predict(df_feature, model, X):
    # === 7. 全体へ予測適用（暴落確率）===
    df_feature['crash_prob'] = model.predict_proba(X)[:, 1]

    # === 8. チャート表示（先頭1万件）===
    df_feature['color'] = df_feature['crash_prob'].apply(lambda x: 'red' if x >= 0.5 else 'deepskyblue')
    sample_df = df_feature#.iloc[:100000]

    plt.figure(figsize=(16, 6))
    plt.scatter(sample_df['jst'], sample_df['mid_price'], c=sample_df['color'], s=1)
    plt.title("Probability Crash（red = High Probability）")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return plt
    

def main():
    symbol = 'NIKKEI'
    nikkei_crash1 = [
        ("2024-07-17 16:00", "2024-07-17 18:00"),
        ("2024-08-01 09:00", "2024-08-05 15:00")
    ]
    
    nikkei_crash2 = [
        ("2024-09-03 22:00", "2024-09-07 00:00")
    ]
    
    nikkei_crash3 = [
        ("2025-03-28 09:00", "2025-04-07 15:00")
    ]
    
    df = read_data('NIKKEI')
    df_feature, X, y = make_features(df, nikkei_crash1)
    model = training(X, y)
    plt = predict(df_feature, model, X)
    plt.savefig(f'./{symbol}_evaluate.png')
    
    
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()