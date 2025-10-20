export class ResetPasswordError extends Error {
  public code: number;
  public phone: string;
  public email: string;

  constructor(params: {
    phone: string;
    email: string;
    code: number;
    message: string;
  }) {
    const { phone, email, code, message } = params;
    super(message);
    this.name = 'ResetPasswordError';
    this.code = code;
    this.phone = phone;
    this.email = email;
  }
}
