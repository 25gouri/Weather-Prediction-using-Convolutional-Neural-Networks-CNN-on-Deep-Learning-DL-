from flask import Flask,session,render_template,request
import mysql.connector
from werkzeug.utils import secure_filename
import os
from prediction import  img_prediction

app=Flask("__name__")
app.secret_key="1234"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/admin_check",methods=["post","get"])
def admin_check():
    uid = request.form['uid']
    pswd = request.form['pwd']

    if uid=="admin" and pswd=="admin":
         return render_template("admin_home.html")
    else:
        return render_template("admin.html",msd="invalid")

@app.route("/ahome")
def ahome():
    return render_template("admin_home.html")

@app.route("/evaluations")
def evaluations():
    return render_template("evaluations.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/uhome")
def uhome():
    return render_template("user_home.html")

@app.route("/user_reg")
def user_reg():
    return render_template("user_reg.html")

@app.route("/user_reg_store",methods =["GET", "POST"])
def user_reg_store():
    name = request.form.get('name')
    uid = request.form.get('uid')
    pwd = request.form.get('pwd')
    email = request.form.get('email')
    mno = request.form.get('mno')
    con, cur = database()
    sql = "select count(*) from users where userid='" + uid + "'"
    cur.execute(sql)
    res = cur.fetchone()[0]
    if res > 0:
        return render_template("user_reg.html", msg="User Id already exists..!")
    else:
        sql = "insert into users values(%s,%s,%s,%s,%s)"
        values = (name, uid, pwd, email, mno)
        cur.execute(sql,values)
        con.commit()
        return render_template("user.html", msg1="Registered Successfully..! Login Here.")
    return ""

@app.route("/user_login_check",methods=['GET','POST'])
def user_login_check():
    uid=request.form.get('uid')
    pswd=request.form.get('pwd')
    con,cur=database()
    sql = "select count(*) from users where userid='" + uid + "' and password='" + pswd + "'"
    cur.execute(sql)
    res = cur.fetchone()[0]
    print("res",res)
    if res > 0:
        session['uid'] = uid

        return render_template("user_home.html")
    else:

        return render_template("user.html", msg="Invalid Credentials")
    return ""

@app.route("/detection")
def detection():
    return render_template("detection.html")

@app.route("/detection2",methods =["GET", "POST"])
def detection2():
    image = request.files['file']
    imgdata = secure_filename(image.filename)
    filename=image.filename
    filelist = [ f for f in os.listdir("testimg") ]
    print(filelist)
    for f in filelist:
        os.remove(os.path.join("testimg", f))
    image.save(os.path.join("testimg", imgdata))
    image_path="..\\satelliteimage\\testimg\\"+filename
    result=img_prediction(image_path)
    con,cur = database()
    qry = "select * from report where image_class='"+result+"' "
    cur.execute(qry)
    res=cur.fetchall()
    print(res)
    return render_template("results.html", res=res)


#DATABASE CONNECTION
def database():
    con = mysql.connector.connect(host="127.0.0.1", user='root', password="root", database="satelliateimage")
    cur = con.cursor()
    return con, cur


if __name__== "__main__" :
    app.run(debug=True,host="localhost",port="2024")
