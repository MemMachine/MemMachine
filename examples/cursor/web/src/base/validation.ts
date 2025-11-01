import { z } from 'zod';

/**
 * Creates password validation rules that enforce:
 * - Minimum length of 8 characters
 * - At least 3 of the following character types:
 *   - Lowercase letters (a-z)
 *   - Uppercase letters (A-Z)
 *   - Numbers (0-9)
 *   - Special characters: ~!@#$%^&*()-_=+\|[{}];:'",<.>/? and space
 */
export const createPasswordValidation = (minLengthMessage: string, complexityMessage: string) => 
  z.string()
    .min(8, {
      message: minLengthMessage,
    })
    .refine((password) => {
      const hasLowercase = /[a-z]/.test(password);
      const hasUppercase = /[A-Z]/.test(password);
      const hasNumber = /\d/.test(password);
      const hasSpecial = /[~!@#$%^&*()\-_=+\\|[\]{};:'",<.>/?\s]/.test(password);
      
      const characterTypes = [hasLowercase, hasUppercase, hasNumber, hasSpecial];
      const validTypes = characterTypes.filter(Boolean).length;
      
      return validTypes >= 3;
    }, {
      message: complexityMessage,
    });

/**
 * Creates username validation rules
 */
export const createUsernameValidation = (message: string) => 
  z.string().min(6, {
    message,
  });

/**
 * Creates password hint validation rules
 */
export const createPasswordHintValidation = (message: string) => 
  z.string().min(1, {
    message,
  });
