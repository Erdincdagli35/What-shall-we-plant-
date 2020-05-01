package com.example.neekelim

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import kotlinx.android.synthetic.main.activity_admin_panel.*

class AdminPanelActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_admin_panel)
        val button = findViewById<Button>(R.id.btnSubmit)
        /*button.setOnClickListener{
            val intent = Intent(this,AdminPanelActivity::class.java)
        */
        val intent = Intent(this,InputActivity::class.java)

        btnSubmit.setOnClickListener {
            if(user_name.text.toString().equals("erdinç") && password.text.toString().equals("123"))
                startActivity(intent)

            else "Hatalı Giris Yapılmıstır"



        }

        }
}
