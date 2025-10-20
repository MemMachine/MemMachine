import { IRespBase } from './common';

export interface IUser {
  id: number;
  username: string;
  email: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface IAuthRespData {
  user: IUser;
  token: string;
  expires_at: string;
}

export interface ILoginReq {
  username: string;
  password: string;
}


export interface IRegisterReq {
  username: string;
  password: string;
  email?: string;
}

export interface IAuthResp extends IRespBase<IAuthRespData> {}