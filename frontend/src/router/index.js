import { createRouter, createWebHistory } from 'vue-router';

const HomeView = () => import('../views/HomeView.vue');

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/faq',
      name: 'faq',
      component: () => import('../views/FAQView.vue'),
    },
    {
      // Informational page on connecting AI agents over MCP. It lives at
      // `/connect` (not `/mcp`) because the MCP transport itself is served
      // same-origin under `/mcp` by the Nginx proxy.
      path: '/connect',
      name: 'connect',
      component: () => import('../views/McpAccess.vue'),
    },
    {
      path: '/verify',
      name: 'verify-email',
      component: () => import('../views/VerifyEmailView.vue'),
    },
    {
      path: '/reset-password',
      name: 'reset-password',
      component: () => import('../views/ResetPasswordView.vue'),
    },
  ],
});

export default router;
