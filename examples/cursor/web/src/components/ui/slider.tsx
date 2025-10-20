
import * as React from 'react';
import * as SliderPrimitive from '@radix-ui/react-slider';

import { cn } from '@/base/utils';

interface SliderProps extends React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root> {
  orientation?: 'horizontal' | 'vertical';
}

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  SliderProps
>(({ className, children, orientation = 'horizontal', ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      'relative flex touch-none select-none items-center',
      orientation === 'vertical' 
        ? 'h-full w-2 flex-col' 
        : 'w-full',
      className
    )}
    orientation={orientation}
    {...props}
  >
    {children ?? (
      <>
        <SliderPrimitive.Track className={cn(
          'relative overflow-hidden rounded-full bg-secondary',
          orientation === 'vertical'
            ? 'h-full w-2 grow'
            : 'h-2 w-full grow'
        )}>
          <SliderPrimitive.Range className={cn(
            'absolute bg-primary',
            orientation === 'vertical'
              ? 'w-full'
              : 'h-full'
          )} />
        </SliderPrimitive.Track>

        <SliderPrimitive.Thumb className="block size-5 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
      </>
    )}
  </SliderPrimitive.Root>
));

Slider.displayName = SliderPrimitive.Root.displayName;

interface SliderTrackProps extends React.ComponentPropsWithoutRef<typeof SliderPrimitive.Track> {
  orientation?: 'horizontal' | 'vertical';
}

const SliderTrack = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Track>,
  SliderTrackProps
>(({ className, orientation = 'horizontal', ...props }, ref) => (
  <SliderPrimitive.Track
    ref={ref}
    className={cn(
      'relative overflow-hidden rounded-full bg-secondary',
      orientation === 'vertical'
        ? 'h-full w-2 grow'
        : 'h-2 w-full grow',
      className
    )}
    {...props}
  />
));

SliderTrack.displayName = SliderPrimitive.Track.displayName;

interface SliderRangeProps extends React.ComponentPropsWithoutRef<typeof SliderPrimitive.Range> {
  orientation?: 'horizontal' | 'vertical';
}

const SliderRange = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Range>,
  SliderRangeProps
>(({ className, orientation = 'horizontal', ...props }, ref) => (
  <SliderPrimitive.Range
    ref={ref}
    className={cn(
      'absolute bg-primary',
      orientation === 'vertical'
        ? 'w-full'
        : 'h-full',
      className
    )}
    {...props}
  />
));

SliderRange.displayName = SliderPrimitive.Range.displayName;

const SliderThumb = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Thumb>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Thumb>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Thumb
    ref={ref}
    className={cn(
      'block size-5 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
      className
    )}
    {...props}
  />
));

SliderThumb.displayName = SliderPrimitive.Thumb.displayName;

export { Slider, SliderTrack, SliderRange, SliderThumb };
