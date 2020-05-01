package com.example.neekelim;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.SQLException;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteException;
import android.database.sqlite.SQLiteOpenHelper;
import android.os.Build;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.Toast;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class database extends SQLiteOpenHelper {

    private static  String dbName = "sqlProduct.db";
    private static String dbPath = "";
    private static  int dbVersion = 2;
    private SQLiteDatabase myDatabase;
    private Context myContext = null;

    public static String inputRegion = "";
    public static String inputpMounth = "";
    public static String inputpTime = "";

    public database(Context context) {
        super(context, dbName, null, dbVersion);
        if(Build.VERSION.SDK_INT>=17)
            dbPath= context.getApplicationInfo().dataDir+"/databases/";
        else
            dbPath = "/data/data/"+ context.getPackageName()+"/databases/";
        myContext=context;

    }

    @Override
    public synchronized void close() {
        if (myDatabase!=null)
            myDatabase.close();
        super.close();
    }

    public boolean checkDatabase(){
        SQLiteDatabase checkDb=null;
        try{
            String myPath= dbPath + dbName;
            checkDb =SQLiteDatabase.openDatabase(myPath,null,SQLiteDatabase.OPEN_READWRITE);
        }
        catch (Exception e){ }
        if(checkDb!=null){
            checkDb.close();
        }
        return  checkDb!=null ? true : false;
    }

    public void copyDatabase(){
        try {
            InputStream myInput = myContext.getAssets().open(dbName);
            String outFileName = dbPath + dbName;
            OutputStream myOutput = new FileOutputStream(outFileName);

            byte [] buffer = new byte[1024];
            int length ;
            while((length=myInput.read(buffer))>0){
                myOutput.write(buffer,0,length);
            }
            myOutput.flush();
            myOutput.close();
            myInput.close();
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }

    public void openDatabase(){
        String myPath = dbPath+dbName;
        myDatabase = SQLiteDatabase.openDatabase(myPath,null,SQLiteDatabase.OPEN_READWRITE);
    }

    public void createDataBase() {
        boolean dbExist = checkDatabase();
        if (dbExist){

        }
        else{
            this.getReadableDatabase();
            try{
                copyDatabase();
            }
            catch (Exception e){

            }
        }

    }


    public StringBuilder  getAllUsers() {
        List<Product> temp = new ArrayList<Product>();
        SQLiteDatabase db = this.getWritableDatabase();

        Product pro = new Product();
        Cursor c = db.rawQuery("Select * From product2 where durum = 'Kar' and " +
                                    "Bölge = '"+inputRegion+"'" +
                                    "and EkildiğiAy ='"+inputpMounth+"'"+
                                    "and ÜretimSüresi = '"+inputpTime+"'"
                                                                ,null);

        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(" - Seçimleriniz  - \n\nBölge : " + inputRegion+" \nEkildiği Ay : "+inputpMounth+" \nÜretim Süresi : "+inputpTime);
        if (c != null) {
            stringBuilder.append("\nÜrün Adı | Üretim | Ekilen Alan | Satış Fiyatı\n\n");
            c.moveToFirst();
            for (int i = 0; i < c.getCount(); i++) {
                    stringBuilder.append((i+1)+" "+c.getString(c.getColumnIndex("Ürün")) + " | "+
                                         c.getString(c.getColumnIndex("Üretim")) + " | "+
                                         c.getString(c.getColumnIndex("EkilenAlan")) + " | "+
                                         c.getString(c.getColumnIndex("SatışFiyatı")) + " | "
                            /*           c.getString(c.getColumnIndex("Bölge")) + " | "+
                                         c.getString(c.getColumnIndex("EkildiğiAy")) + " | "+
                                         c.getString(c.getColumnIndex("ÜretimSüresi")) + " | "
                                         /*c.getString(c.getColumnIndex("binarySonuc"))*/);
                stringBuilder.append("\n");
                c.moveToNext();
            }
            c.close();
            //c.close();
            //db.close();
        }
        return stringBuilder;
        //Cursor c = db.query("account",columNS,null,null,null,null,null);
        //looping through all the records
        /*int count = 0;
        while (c.moveToNext()) {
            Account a = new Account();
            a.setUserName(c.getString(c.getColumnIndex("username")));
            a.setPassword(c.getString(c.getColumnIndex("password")));
            count++;
            temp.add(a);

            if (c.getPosition()==0) {

                c.moveToPosition(position);
            }
        }

         */
    }
    @Override
    public void onCreate(SQLiteDatabase db) {
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.setVersion(oldVersion);
    }



}
