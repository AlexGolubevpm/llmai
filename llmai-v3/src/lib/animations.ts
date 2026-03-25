import type { Variants, Transition } from "framer-motion";

export const spring: Transition = {
  type: "spring" as const,
  stiffness: 300,
  damping: 30,
};

// Page-level animations
export const pageVariants = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  exit: { opacity: 0, y: -4, transition: { duration: 0.2 } },
};

// Staggered children
export const staggerContainer = {
  animate: {
    transition: { staggerChildren: 0.05 },
  },
};

export const staggerItem = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3 } },
};

// Card animations
export const cardVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

// Fade in
export const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.2 } },
  exit: { opacity: 0, transition: { duration: 0.15 } },
};

// Slide from right (for sheets/panels)
export const slideRight = {
  initial: { x: "100%" },
  animate: { x: 0, transition: spring },
  exit: { x: "100%", transition: { duration: 0.2 } },
};

// Scale (for modals)
export const scaleIn = {
  initial: { opacity: 0, scale: 0.95 },
  animate: { opacity: 1, scale: 1, transition: spring },
  exit: { opacity: 0, scale: 0.95, transition: { duration: 0.15 } },
};

// Table row
export const tableRowVariants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.2 } },
  exit: { opacity: 0, x: -20, transition: { duration: 0.2 } },
};
