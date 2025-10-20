import { AxiosError, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios'

import { handleAPIError, MemMachineAPIError } from '@/errors'

describe('Handle API Error', () => {
  it('should throw MemMachineAPIError with message from AxiosError', () => {
    const mockAxiosResponse: AxiosResponse = {
      data: { detail: 'Extra detail' },
      status: 400,
      statusText: 'Bad Request',
      headers: {},
      config: { headers: {} } as InternalAxiosRequestConfig
    }
    const axiosError = new AxiosError(
      'Axios error',
      'ERR_BAD_REQUEST',
      undefined,
      undefined,
      mockAxiosResponse
    )
    expect(() => handleAPIError(axiosError, 'Context')).toThrow(MemMachineAPIError)
    expect(() => handleAPIError(axiosError, 'Context')).toThrow('Context: Axios error - Extra detail')
  })

  it('should throw MemMachineAPIError with message from Error', () => {
    const error = new Error('Test error')
    expect(() => handleAPIError(error, 'Context')).toThrow(MemMachineAPIError)
    expect(() => handleAPIError(error, 'Context')).toThrow('Context: Test error')
  })

  it('should throw MemMachineAPIError with stringified error for other error', () => {
    const errorObj = { foo: 'bar' }
    expect(() => handleAPIError(errorObj, 'Context')).toThrow(MemMachineAPIError)
    expect(() => handleAPIError(errorObj, 'Context')).toThrow('Context: {"foo":"bar"}')
  })
})
