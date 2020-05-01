package com.example.neekelim

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {


    override fun onCreate( savedInstanceState: Bundle?)
    {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_main)

        val button = findViewById<Button>(R.id.adminBtn)
        val buttonF = findViewById<Button>(R.id.farmerBtn)
        button.setOnClickListener{
            val intent = Intent(this,AdminpActivity::class.java)

            startActivity(intent)
        }
        buttonF.setOnClickListener{
            val intent = Intent(this,InputActivity::class.java)

            startActivity(intent)
        }
    }
}
