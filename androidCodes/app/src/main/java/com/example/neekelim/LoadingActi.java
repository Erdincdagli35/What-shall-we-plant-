package com.example.neekelim;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import androidx.appcompat.app.AppCompatActivity;
import java.util.HashMap;
import kotlin.Metadata;
import org.jetbrains.annotations.Nullable;

public class LoadingActi extends AppCompatActivity {
    private int SPLASH_TIME = 3000;
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.setContentView(R.layout.activity_loading);
        (new Handler()).postDelayed((Runnable)(new Runnable() {
            public final void run() {
                Intent mySuperIntent = new Intent((Context)LoadingActi.this, MainActi.class);
                LoadingActi.this.startActivity(mySuperIntent);
                LoadingActi.this.finish();
            }
        }), (long)this.SPLASH_TIME);
    }

}
