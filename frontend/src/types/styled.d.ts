import 'styled-components';
import { CyberpunkTheme } from '../theme/cyberpunkTheme';

declare module 'styled-components' {
  export interface DefaultTheme extends CyberpunkTheme {}
}