FROM ubuntu

RUN apt-get update -y

RUN apt-get install ruby-full build-essential zlib1g-dev -y && \
    echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc && \
    echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc && \
    echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc

WORKDIR /site

COPY . .

RUN /bin/bundle3.0 install && /bin/gem install jekyll bundler && /bin/bundle3.0 add webrick


CMD [ "bundle", "exec", "jekyll serve --host 0.0.0.0" ]

EXPOSE 22
EXPOSE 4000