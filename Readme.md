# Charles's Personal Site

Built with Jekyll using based on [Willian Justen's template](https://github.com/willianjusten/will-jekyll-template)

## Getting Started

To work on the site, use **docker-compose**. Run `docker-compose up` and the jekyll site will be served at `localhost:3000` with automated refresh.

Before working on the site, remember to change `url` in `config.yaml` to `localhost:3000`

## Images

Put images into `src/img`. Check that the image has been copied to `assets/`. If not, run `imagemin`:
```bash
docker-compose up -d
# < Move image >
docker-compose exec node /bin/bash
./node_modules/gulp/bin/gulp.js imagemin
```
