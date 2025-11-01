import { IRequest } from '@/base/createRequest';
import { request } from '@/base/request';
import type { ILoginReq, IRegisterReq, IAuthResp } from '@/models/user';

class UserService {
  constructor(private request: IRequest) {}

  public login = async (body: ILoginReq) => {
    return this.request<IAuthResp>({
      url: '/api/auth/login',
      method: 'POST',
      redirectUnauthorized: false,
      body,
    });
  };

  public logout = async () => {
    return this.request<any>({
      url: '/api/auth/logout',
      method: 'POST',
    });
  };

  public register = async (body: IRegisterReq) => {
    return this.request<IAuthResp>({
      url: '/api/auth/register',
      method: 'POST',
      redirectUnauthorized: false,
      body,
    });
  };

  public getCurrentUser = async () => {
    return this.request<IAuthResp>({
      url: '/api/auth/me',
      method: 'GET',
    });
  };
}
export const userService = new UserService(request);
