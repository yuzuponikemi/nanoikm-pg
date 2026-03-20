"""
Vulnerable Web App — 練習専用

⚠️  このアプリは教育目的で意図的に脆弱に作られています。
    絶対にインターネットに公開しないでください。
    ローカル環境（localhost）でのみ使用してください。

含まれる脆弱性:
  - SQLインジェクション (Lab 1, 2)
  - XSS クロスサイトスクリプティング (Lab 3)
  - 安全でない認証 (Lab 4)
"""

import sqlite3
import os
from flask import Flask, request, render_template_string, session, redirect, url_for, g

app = Flask(__name__)
app.secret_key = "super_secret_123"  # 脆弱性: 弱いシークレットキー

DB_PATH = "lab.db"


# ──────────────────────────────────────────
# データベース初期化
# ──────────────────────────────────────────

def init_db():
    """練習用データベースを初期化する"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ユーザーテーブル
    c.execute("DROP TABLE IF EXISTS users")
    c.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            password TEXT,
            role TEXT
        )
    """)
    c.execute("INSERT INTO users VALUES (1, 'alice', 'password123', 'user')")
    c.execute("INSERT INTO users VALUES (2, 'bob', 'qwerty', 'user')")
    c.execute("INSERT INTO users VALUES (3, 'admin', 'admin_secret_pw', 'admin')")

    # 商品テーブル（SQLiの練習用）
    c.execute("DROP TABLE IF EXISTS products")
    c.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price INTEGER,
            category TEXT
        )
    """)
    c.execute("INSERT INTO products VALUES (1, 'リンゴ', 100, 'fruit')")
    c.execute("INSERT INTO products VALUES (2, 'バナナ', 80, 'fruit')")
    c.execute("INSERT INTO products VALUES (3, 'ニンジン', 120, 'vegetable')")
    c.execute("INSERT INTO products VALUES (4, '秘密の商品', 9999, 'secret')")

    # フラグテーブル（CTF的な達成確認用）
    c.execute("DROP TABLE IF EXISTS flags")
    c.execute("""
        CREATE TABLE flags (
            id INTEGER PRIMARY KEY,
            flag TEXT,
            hint TEXT
        )
    """)
    c.execute("INSERT INTO flags VALUES (1, 'FLAG{sql_injection_master}', 'SQLiでこのテーブルを見つけよう')")

    conn.commit()
    conn.close()


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ──────────────────────────────────────────
# トップページ
# ──────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>脆弱アプリ - 練習場</title>
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
    .warning { background: #fff3cd; border: 1px solid #ffc107; padding: 12px; border-radius: 6px; }
    .lab { background: #f8f9fa; border: 1px solid #dee2e6; padding: 16px; margin: 12px 0; border-radius: 6px; }
    a { color: #0066cc; }
    h1 { color: #cc0000; }
  </style>
</head>
<body>
  <h1>⚠️ 脆弱アプリ（練習専用）</h1>
  <div class="warning">
    <strong>注意:</strong> このアプリは教育目的で意図的に脆弱に作られています。
    localhost のみで使用してください。
  </div>

  <h2>ラボ一覧</h2>

  <div class="lab">
    <h3>Lab 1: SQLインジェクション — 商品検索</h3>
    <p>検索フォームにSQL文を注入してみよう</p>
    <a href="/lab/sqli/search">→ 商品検索ページへ</a>
  </div>

  <div class="lab">
    <h3>Lab 2: SQLインジェクション — ログイン bypass</h3>
    <p>パスワードを知らずにログインできるか？</p>
    <a href="/lab/sqli/login">→ ログインページへ</a>
  </div>

  <div class="lab">
    <h3>Lab 3: XSS — コメント欄</h3>
    <p>コメントにJavaScriptを埋め込んでみよう</p>
    <a href="/lab/xss/comment">→ コメントページへ</a>
  </div>

  <div class="lab">
    <h3>Lab 4: 安全なバージョン</h3>
    <p>修正後のコードと比較してみよう</p>
    <a href="/lab/safe/search">→ 安全な検索ページへ</a>
  </div>

  <div class="lab">
    <h3>Lab 5: 認証の脆弱性 — 弱いセッション管理</h3>
    <p>推測可能なセッションIDの問題を体験しよう</p>
    <a href="/lab/auth/login">→ 認証ラボへ</a>
  </div>
</body>
</html>
""")


# ──────────────────────────────────────────
# Lab 1: SQLインジェクション — 商品検索
# ──────────────────────────────────────────

@app.route("/lab/sqli/search")
def sqli_search():
    query = request.args.get("q", "")
    results = []
    error = None
    sql_shown = ""

    if query:
        # 🚨 脆弱: ユーザー入力をそのままSQLに埋め込んでいる
        sql_shown = f"SELECT * FROM products WHERE name LIKE '%{query}%'"
        try:
            db = get_db()
            results = db.execute(sql_shown).fetchall()
        except Exception as e:
            error = str(e)

    return render_template_string("""
<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><title>Lab 1: SQLi 商品検索</title>
<style>
  body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
  .sql { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 4px; font-family: monospace; }
  .hint { background: #e8f4f8; padding: 10px; border-left: 4px solid #0066cc; margin: 10px 0; }
  table { width: 100%; border-collapse: collapse; }
  th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
  th { background: #f0f0f0; }
  .error { color: red; }
</style>
</head>
<body>
  <h1>Lab 1: SQLインジェクション — 商品検索</h1>
  <p><a href="/">← 戻る</a></p>

  <form method="get">
    <input name="q" value="{{ query }}" placeholder="商品名を検索..." style="width:300px;padding:8px;">
    <button type="submit">検索</button>
  </form>

  <h3>実行されるSQL:</h3>
  <div class="sql">{{ sql_shown if sql_shown else "（検索後に表示）" }}</div>

  <div class="hint">
    💡 ヒント: 通常の検索は「リンゴ」のように入力します。<br>
    攻撃例として <code>%' OR '1'='1</code> を試してみましょう。
  </div>

  {% if error %}
    <p class="error">エラー: {{ error }}</p>
  {% endif %}

  {% if results %}
  <h3>検索結果 ({{ results|length }} 件):</h3>
  <table>
    <tr><th>ID</th><th>商品名</th><th>価格</th><th>カテゴリ</th></tr>
    {% for row in results %}
    <tr><td>{{ row[0] }}</td><td>{{ row[1] }}</td><td>{{ row[2] }}</td><td>{{ row[3] }}</td></tr>
    {% endfor %}
  </table>
  {% endif %}
</body></html>
""", query=query, sql_shown=sql_shown, results=results, error=error)


# ──────────────────────────────────────────
# Lab 2: SQLインジェクション — ログインbypass
# ──────────────────────────────────────────

@app.route("/lab/sqli/login", methods=["GET", "POST"])
def sqli_login():
    message = ""
    sql_shown = ""

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        # 🚨 脆弱: 文字列結合でSQL構築
        sql_shown = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        try:
            db = get_db()
            user = db.execute(sql_shown).fetchone()
            if user:
                message = f"✅ ログイン成功！ ようこそ {user['username']} さん（権限: {user['role']}）"
            else:
                message = "❌ ユーザー名またはパスワードが違います"
        except Exception as e:
            message = f"エラー: {e}"

    return render_template_string("""
<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><title>Lab 2: SQLi Login</title>
<style>
  body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
  .sql { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 4px; font-family: monospace; word-break: break-all; }
  .hint { background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }
  input { display: block; width: 100%; padding: 8px; margin: 6px 0 12px; box-sizing: border-box; }
  button { padding: 10px 20px; background: #0066cc; color: white; border: none; cursor: pointer; }
  .success { color: green; font-weight: bold; }
  .fail { color: red; }
</style>
</head>
<body>
  <h1>Lab 2: ログインbypass</h1>
  <p><a href="/">← 戻る</a></p>

  <form method="post">
    <label>ユーザー名</label>
    <input name="username" placeholder="alice">
    <label>パスワード</label>
    <input name="password" type="text" placeholder="（パスワードを知らなくてもログインできる？）">
    <button type="submit">ログイン</button>
  </form>

  {% if sql_shown %}
  <h3>実行されたSQL:</h3>
  <div class="sql">{{ sql_shown }}</div>
  {% endif %}

  {% if message %}
  <p class="{{ 'success' if '成功' in message else 'fail' }}">{{ message }}</p>
  {% endif %}

  <div class="hint">
    💡 ヒント: ユーザー名に <code>admin' --</code> を入力するとどうなる？<br>
    <code>--</code> はSQLのコメント記号です。
  </div>
</body></html>
""", message=message, sql_shown=sql_shown)


# ──────────────────────────────────────────
# Lab 3: XSS — コメント欄
# ──────────────────────────────────────────

comments = []  # メモリ上に保持（永続化しない）

@app.route("/lab/xss/comment", methods=["GET", "POST"])
def xss_comment():
    if request.method == "POST":
        name = request.form.get("name", "匿名")
        comment = request.form.get("comment", "")
        comments.append({"name": name, "comment": comment})

    # 🚨 脆弱: |safe フィルターでエスケープを無効化
    comments_html = ""
    for c in comments:
        comments_html += f'<div class="comment"><b>{c["name"]}</b>: {c["comment"]}</div>'

    return render_template_string("""
<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><title>Lab 3: XSS</title>
<style>
  body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
  .comment { background: #f8f9fa; border: 1px solid #ddd; padding: 10px; margin: 8px 0; border-radius: 4px; }
  .hint { background: #f8d7da; padding: 10px; border-left: 4px solid #dc3545; margin: 10px 0; }
  input, textarea { width: 100%; padding: 8px; margin: 4px 0 12px; box-sizing: border-box; }
  button { padding: 10px 20px; background: #28a745; color: white; border: none; cursor: pointer; }
</style>
</head>
<body>
  <h1>Lab 3: XSS — コメント欄</h1>
  <p><a href="/">← 戻る</a></p>

  <form method="post">
    <label>名前</label>
    <input name="name" placeholder="あなたの名前">
    <label>コメント</label>
    <textarea name="comment" rows="3" placeholder="コメントを入力..."></textarea>
    <button type="submit">投稿</button>
  </form>

  <div class="hint">
    💡 攻撃例: コメントに <code>&lt;script&gt;alert('XSS!')&lt;/script&gt;</code> を入力してみよう
  </div>

  <h3>コメント一覧:</h3>
  """ + comments_html + """
</body></html>
""")


# ──────────────────────────────────────────
# Lab 4: 安全なバージョン（比較用）
# ──────────────────────────────────────────

@app.route("/lab/safe/search")
def safe_search():
    query = request.args.get("q", "")
    results = []
    sql_shown = ""

    if query:
        # ✅ 安全: プレースホルダー（パラメータ化クエリ）を使用
        sql_shown = f"SELECT * FROM products WHERE name LIKE ? → パラメータ: ['%{query}%']"
        db = get_db()
        results = db.execute(
            "SELECT * FROM products WHERE name LIKE ?",
            (f"%{query}%",)
        ).fetchall()

    return render_template_string("""
<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><title>安全な検索</title>
<style>
  body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
  .sql { background: #1e3a1e; color: #90ee90; padding: 12px; border-radius: 4px; font-family: monospace; }
  .safe { background: #d4edda; padding: 10px; border-left: 4px solid #28a745; margin: 10px 0; }
  table { width: 100%; border-collapse: collapse; }
  th, td { border: 1px solid #ccc; padding: 8px; }
  th { background: #f0f0f0; }
</style>
</head>
<body>
  <h1>✅ 安全な検索（比較用）</h1>
  <p><a href="/">← 戻る</a></p>

  <div class="safe">
    このページはパラメータ化クエリを使っているため、SQLインジェクションが効きません。
    Lab 1 と同じ攻撃文字列を試してみてください！
  </div>

  <form method="get">
    <input name="q" value="{{ query }}" placeholder="商品名を検索..." style="width:300px;padding:8px;">
    <button type="submit">検索</button>
  </form>

  <h3>実行されるSQL（安全）:</h3>
  <div class="sql">{{ sql_shown if sql_shown else "（検索後に表示）" }}</div>

  {% if results %}
  <h3>検索結果 ({{ results|length }} 件):</h3>
  <table>
    <tr><th>ID</th><th>商品名</th><th>価格</th><th>カテゴリ</th></tr>
    {% for row in results %}
    <tr><td>{{ row[0] }}</td><td>{{ row[1] }}</td><td>{{ row[2] }}</td><td>{{ row[3] }}</td></tr>
    {% endfor %}
  </table>
  {% endif %}
</body></html>
""", query=query, sql_shown=sql_shown, results=results)


# ──────────────────────────────────────────
# Lab 5: 認証の脆弱性 — 弱いセッション管理
# ──────────────────────────────────────────

auth_sessions = {}  # 脆弱なセッションストア（メモリ上）
auth_session_counter = 0  # 🚨 脆弱: 連番のセッションID

@app.route("/lab/auth/login", methods=["GET", "POST"])
def auth_login():
    global auth_session_counter
    message = ""
    current_user = None

    # セッションの確認
    session_id = request.cookies.get("lab_session")
    if session_id and session_id in auth_sessions:
        current_user = auth_sessions[session_id]

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "login":
            username = request.form.get("username", "")
            password = request.form.get("password", "")

            # 簡易ユーザーDB
            users = {"alice": "password123", "bob": "qwerty", "admin": "admin_secret_pw"}

            if username in users and users[username] == password:
                # 🚨 脆弱: 連番のセッションID
                auth_session_counter += 1
                new_session_id = str(auth_session_counter)
                auth_sessions[new_session_id] = {"username": username, "role": "admin" if username == "admin" else "user"}

                resp = redirect(url_for("auth_login"))
                # 🚨 脆弱: HttpOnly なし、Secure なし
                resp.set_cookie("lab_session", new_session_id)
                return resp
            else:
                message = "❌ ユーザー名またはパスワードが違います"

        elif action == "logout":
            if session_id and session_id in auth_sessions:
                del auth_sessions[session_id]
            resp = redirect(url_for("auth_login"))
            resp.delete_cookie("lab_session")
            return resp

    return render_template_string("""
<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><title>Lab 5: 認証</title>
<style>
  body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
  .hint { background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }
  .vuln { background: #f8d7da; padding: 10px; border-left: 4px solid #dc3545; margin: 10px 0; }
  .info { background: #e8f4f8; padding: 10px; border-left: 4px solid #0066cc; margin: 10px 0; }
  input { display: block; width: 100%; padding: 8px; margin: 6px 0 12px; box-sizing: border-box; }
  button { padding: 10px 20px; background: #0066cc; color: white; border: none; cursor: pointer; margin: 4px; }
  .success { color: green; font-weight: bold; }
  .fail { color: red; }
</style>
</head>
<body>
  <h1>Lab 5: 認証の脆弱性</h1>
  <p><a href="/">← 戻る</a></p>

  {% if current_user %}
  <div class="info">
    <p>ログイン中: <strong>{{ current_user.username }}</strong>（権限: {{ current_user.role }}）</p>
    <p>セッションID: <code>{{ session_id }}</code></p>
    <form method="post" style="display:inline">
      <input type="hidden" name="action" value="logout">
      <button type="submit" style="background:#dc3545;">ログアウト</button>
    </form>
  </div>

  <div class="vuln">
    <h3>この認証の脆弱性:</h3>
    <ul>
      <li>セッションIDが連番（{{ session_id }}） → 他のユーザーのIDを推測可能</li>
      <li>Cookie に HttpOnly が未設定 → XSS で盗取可能</li>
      <li>Cookie に Secure が未設定 → HTTP通信で盗聴可能</li>
      <li>パスワードが平文で保存 → DB漏洩で全パスワード流出</li>
    </ul>
  </div>

  {% else %}

  <form method="post">
    <input type="hidden" name="action" value="login">
    <label>ユーザー名</label>
    <input name="username" placeholder="alice">
    <label>パスワード</label>
    <input name="password" type="text" placeholder="password123">
    <button type="submit">ログイン</button>
  </form>

  {% if message %}
  <p class="fail">{{ message }}</p>
  {% endif %}

  {% endif %}

  <div class="hint">
    <h3>攻撃シナリオ</h3>
    <ol>
      <li>alice でログインする（password123）</li>
      <li>Cookie のセッションID（数字）を確認する</li>
      <li>別のブラウザ/シークレットウィンドウで Cookie を手動セット</li>
      <li>セッションIDを 1 ずつ変えて他ユーザーになりすまし</li>
    </ol>
    <p>ブラウザの開発者ツール → Application → Cookies で確認できます</p>
  </div>
</body></html>
""", current_user=current_user, session_id=session_id, message=message)


# ──────────────────────────────────────────
# 起動
# ──────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        init_db()
        print("データベースを初期化しました")
    else:
        init_db()  # 毎回リセット（練習のため）
    print("\n⚠️  練習専用アプリを起動します")
    print("   http://localhost:5000 にアクセスしてください")
    print("   Ctrl+C で停止\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
