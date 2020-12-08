import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  _HOST = "http://localhost:5000/";
  constructor(private  _HTTP:  HttpClient) { }
  /** Get all images from database */
  public getAllRequests() {
    let headers 		: any		 = new HttpHeaders({ 'Content-Type': 'application/json' ,'Access-Control-Allow-Origin': '*'}),
    url       	: any      	 = this._HOST;
    return this._HTTP.get(url, headers)
  }
   /** Register new user  */
  public createNewRequest(formObj: any)
  {


     const formData = new FormData();
     formData.append('file', formObj.get('file').value);
     formData.append('category', formObj.get('category').value);




     let headers 		: any		 = new HttpHeaders({ 'Content-Type': 'multipart/form-data','Accept':'application/json','Access-Control-Allow-Origin': '*' }),
    //options 		: any 		 = { file : formObj.get('file').value, target : formObj.get('target').value},
    url       	: any      	 = this._HOST + "anomaly";

        return this._HTTP.post(url, formObj, headers);

  }
  public getStatus(){
     let headers 		: any		 = new HttpHeaders({ 'Content-Type': 'application/json' ,'Access-Control-Allow-Origin': '*'}),
    url       	: any      	 =  this._HOST + 'status';
    return this._HTTP.get(url, headers)

  }
  public testAPI(){
     let headers 		: any		 = new HttpHeaders({ 'Content-Type': 'application/json' ,'Access-Control-Allow-Origin': '*'}),
    url       	: any      	 =  this._HOST + 'debug';
    return this._HTTP.get(url, headers)

  }
}
