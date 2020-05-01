package com.example.neekelim;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import com.example.neekelim.R.id;
import java.util.HashMap;
import kotlin.Metadata;
import kotlin.jvm.internal.Intrinsics;
import org.jetbrains.annotations.Nullable;


public class AdminpActivity extends AppCompatActivity{
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_admin_panel);
        Button button = (Button)this.findViewById(id.btnSubmit);
        EditText user_name = (EditText) findViewById(id.user_name);
        EditText password = (EditText) findViewById(id.password);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(user_name.getText().toString().equals("erdinç") && password.getText().toString().equals("123")){
                    Intent intent = new Intent(getApplicationContext(), InputActivity.class);
                    startActivity(intent);
                }
                else{
                    Toast.makeText(AdminpActivity.this, "Hatalı Giris Yapılmıstır", Toast.LENGTH_SHORT).show();
                }

            }
        });
    }
}
