package com.example.neekelim;

public class Product {
    String productName,pMounth,region,pTime,state,m2;
    int production,productionTime,price;

    public Product( String m2, String region,String pMounth, String pTime) {
        this.pMounth = pMounth;
        this.region = region;
        this.pTime = pTime;
        this.m2 = m2;
    }

    public Product() {

    }

    public Product(String productName, String pMounth, String region, String pTime, String state, int production, int productionTime, int price) {
        this.productName = productName;
        this.pMounth = pMounth;
        this.region = region;
        this.pTime = pTime;
        this.state = state;
        this.production = production;
        this.productionTime = productionTime;
        this.price = price;
    }


    public String getM2() {
        return m2;
    }
    public void setM2(String m2) {
        this.m2 = m2;
    }

    public String getProductName() {
        return productName;
    }

    public void setProductName(String productName) {
        this.productName = productName;
    }

    public String getpMounth() {
        return pMounth;
    }

    public void setpMounth(String pMounth) {
        this.pMounth = pMounth;
    }

    public String getRegion() {
        return region;
    }

    public void setRegion(String region) {
        this.region = region;
    }

    public String getpTime() {
        return pTime;
    }

    public void setpTime(String pTime) {
        this.pTime = pTime;
    }

    public String getState() {
        return state;
    }

    public void setState(String state) {
        this.state = state;
    }

    public int getProduction() {
        return production;
    }

    public void setProduction(int production) {
        this.production = production;
    }

    public int getProductionTime() {
        return productionTime;
    }

    public void setProductionTime(int productionTime) {
        this.productionTime = productionTime;
    }

    public int getPrice() {
        return price;
    }

    public void setPrice(int price) {
        this.price = price;
    }
}
