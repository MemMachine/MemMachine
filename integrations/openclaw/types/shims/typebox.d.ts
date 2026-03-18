declare module "@sinclair/typebox" {
  export const Type: {
    Object: (...args: any[]) => any;
    String: (...args: any[]) => any;
    Number: (...args: any[]) => any;
    Boolean: (...args: any[]) => any;
    Optional: (...args: any[]) => any;
    Union: (...args: any[]) => any;
    Literal: (...args: any[]) => any;
    Array: (...args: any[]) => any;
    Record: (...args: any[]) => any;
  };
}
