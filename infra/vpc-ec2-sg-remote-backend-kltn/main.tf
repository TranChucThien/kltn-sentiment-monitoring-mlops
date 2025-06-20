provider "aws" {
  region = "us-east-2"

}


terraform {
  

  backend "s3" {
  bucket         = "tranchucthienops"
  key            = "spark_env/terraform.tfstate"
  region         = "us-east-2"
  use_lockfile   = true
  encrypt = true
}

  required_version = ">= 1.0.0"
}
# module "vpc" {
#   source = "./modules/vpc"
#   name   = "tct-vpc"

# }

module "ec2" {
  source   = "./modules/ec2-sg" # Path to the module directory
  key_name = "thien"
  ec2_name = "spark_terra"   # Example tag name for EC2 instance, can be overridden
  key_path = "./emr.pem" # Path to the private key file
  instance_ami = "ami-08611213135bc447a"
  instance_type = "t2.2xlarge" # 
  ingress_rules = [
    { from_port = 22, to_port = 22, protocol = "tcp", cidr_blocks = ["0.0.0.0/0"] },   # SSH
    { from_port = 80, to_port = 80, protocol = "tcp", cidr_blocks = ["0.0.0.0/0"] },   # HTTP
    { from_port = 443, to_port = 443, protocol = "tcp", cidr_blocks = ["0.0.0.0/0"] }, # HTTPS
  ]
  sg_description = "SG with SSH and HTTP/HTTPS access"
  vpc_id         = "vpc-0cec5035ecf031d0b"
  subnet_id      = "subnet-023a104aea233df54"
  # provisioner_commands = [
  #   "ls"
  # ]
}

