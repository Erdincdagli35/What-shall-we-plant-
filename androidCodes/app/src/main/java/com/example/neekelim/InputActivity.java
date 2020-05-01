package com.example.neekelim;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Menu;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import java.util.ArrayList;
import java.util.List;

public class InputActivity extends AppCompatActivity {
    private Spinner spinnerpTimesItems;
    private Spinner spinnerpMounth;
    private Spinner spinnerpRegion;

    ArrayAdapter<String> arrayAdapter_pTime,arrayAdapter_pMounth,arrayAdapter_pRegion;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_input);

        //Spinner spinner =(Spinner)findViewById(R.id.spinnerRegion);
        EditText txtM2 = (EditText) findViewById(R.id.m2);
        /*EditText txtRegion = (EditText) findViewById(R.id.region);
        EditText txtPMounth = (EditText) findViewById(R.id.pMounth);
        EditText txtPTime = (EditText) findViewById(R.id.pTime);*/
        Button btnSubmt = (Button) findViewById(R.id.btnSubmit);


        spinnerpTimesItems = findViewById(R.id.pTimeSpinner);
        spinnerpMounth=findViewById(R.id.spinnerpMounth);
        spinnerpRegion=findViewById(R.id.spinnerpRegion);

        String[] pTimesItems={"Üretim Süresini Seçin","Kısa Süre","Uzun Süre "};
        String[] pMounthItems={"Ekilecek Ayı Seçin","Ocak","Şubat","Mart","Nisan","Mayıs","Haziran","Temmuz","Ağustos","Eylül","Ekim","Kasım","Aralık"};
        String[] pRegionItems={"Bölgeyi Seçin","Akdeniz","Doğu Anadolu","Ege","Güneydoğu Anadolu","Karadeniz","Marmara","İç Anadolu"};

        arrayAdapter_pTime=new ArrayAdapter<String>(getApplicationContext(),android.R.layout.simple_spinner_item, pTimesItems)
        {
            @Override
            public boolean isEnabled(int position){
                if(position == 0)
                {
                    // Disable the first item from Spinner
                    // First item will be use for hint
                    return false;
                }
                else
                {
                    return true;
                }
            }
            @Override
            public View getDropDownView(int position, View convertView,
                                        ViewGroup parent) {
                View view = super.getDropDownView(position, convertView, parent);
                TextView tv = (TextView) view;
                if(position == 0){
                    // Set the hint text color gray
                    tv.setTextColor(Color.WHITE);
                }
                else {
                    tv.setTextColor(Color.GREEN);
                }
                return view;
            }
        };
        spinnerpTimesItems.setAdapter(arrayAdapter_pTime);

        arrayAdapter_pMounth=new ArrayAdapter<String>(getApplicationContext(),android.R.layout.simple_spinner_item, pMounthItems){
            @Override
            public boolean isEnabled(int position){
                if(position == 0)
                {
                    // Disable the first item from Spinner
                    // First item will be use for hint
                    return false;
                }
                else
                {
                    return true;
                }
            }
            @Override
            public View getDropDownView(int position, View convertView,
                                        ViewGroup parent) {
                View view = super.getDropDownView(position, convertView, parent);
                TextView tv = (TextView) view;
                if(position == 0){
                    // Set the hint text color gray
                    tv.setTextColor(Color.WHITE);
                }
                else {
                    tv.setTextColor(Color.GREEN);
                }
                return view;
            }
        };
        spinnerpMounth.setAdapter(arrayAdapter_pMounth);

        arrayAdapter_pRegion=new ArrayAdapter<String>(getApplicationContext(),android.R.layout.simple_spinner_item, pRegionItems){
            @Override
            public boolean isEnabled(int position){
                if(position == 0)
                {
                    // Disable the first item from Spinner
                    // First item will be use for hint
                    return false;
                }
                else
                {
                    return true;
                }
            }
            @Override
            public View getDropDownView(int position, View convertView,
                                        ViewGroup parent) {
                View view = super.getDropDownView(position, convertView, parent);
                TextView tv = (TextView) view;
                if(position == 0){
                    // Set the hint text color gray
                    tv.setTextColor(Color.WHITE);
                }
                else {
                    tv.setTextColor(Color.GREEN);
                }
                return view;
            }
        };;
        spinnerpRegion.setAdapter(arrayAdapter_pRegion);

        database db = new database(getApplicationContext());

        spinnerpTimesItems.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            String pTimesItemsAssign;
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if(position > 0){
                    //Toast.makeText(getApplicationContext(), "Üretim Süresi : "+pTimesItems[position]+" \nSeçildi" ,Toast.LENGTH_SHORT).show();
                    pTimesItemsAssign=pTimesItems[position];
                    db.inputpTime=pTimesItemsAssign;
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        spinnerpRegion.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            String pTimesItemsAssign;
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if(position > 0) {
                    //Toast.makeText(getApplicationContext(), "Bölge : " + pRegionItems[position] + " \nSeçildi", Toast.LENGTH_SHORT).show();
                    pTimesItemsAssign = pRegionItems[position];
                    db.inputRegion = pTimesItemsAssign;
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        spinnerpMounth.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            String pTimesItemsAssign;
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if(position > 0) {
                    //Toast.makeText(getApplicationContext(), "Ekildiği Ay : " + pMounthItems[position] + " \nSeçildi", Toast.LENGTH_SHORT).show();
                    pTimesItemsAssign = pMounthItems[position];
                    db.inputpMounth = pTimesItemsAssign;
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        btnSubmt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                String m2 = txtM2.getText().toString();
                /*String region = txtRegion.getText().toString();
                String pMounth = txtPMounth.getText().toString();
                String pTime2 = txtPTime.getText().toString();
                *///selectedRegion=region;

                //db.inputpTime=pTime(pt);

                if(m2.isEmpty()){
                    Toast.makeText(getApplicationContext(), "Alanları Boş Bırakmayınız", Toast.LENGTH_SHORT).show();
                    return ;
                }
                else {

                    Intent intent = new Intent(getApplicationContext(), ResultActivity.class);
                    startActivity(intent);

                }

        };
            });
    }
}