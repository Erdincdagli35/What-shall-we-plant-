package com.example.neekelim;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Context;
import android.database.Cursor;
import android.database.SQLException;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class ResultActivity extends AppCompatActivity{

    List<Product> lstUser = new ArrayList<Product>();
    database db ;
    Button btnGetData;
    LinearLayout container;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
        //btnGetData = (Button) findViewById(R.id.button);
        EditText containerT = (EditText) findViewById(R.id.editText3);

        db = new database(getApplicationContext());
        db.createDataBase();

        db.getAllUsers();

        containerT.append(db.getAllUsers());

        /*btnGetData.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

            }
        });
        */


    }

}
/*StringBuilder sb = new StringBuilder();
                for(Product product : lstUser){
                    //LayoutInflater inflater =(LayoutInflater)getBaseContext().getSystemService(Context.LAYOUT_INFLATER_SERVICE);
                    //View addView = inflater.inflate(R.layout.row,null);
                    //TextView txtUser = (TextView) addView.findViewById(R.id.txtUser);
                    //TextView txtPassword = (TextView) addView.findViewById(R.id.txtPassword);

                    //txtUser.setText(acc.getUserName());
                    //txtPassword.setText(acc.getPassword());
                    String content = "";
                    content =product.getProductName()+"\n";
                    sb.append(content);
                }*/